import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
import streamlit as st
import matplotlib.pyplot as plt
import pennylane as qml
import tempfile

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = r"BrainTumorUI/hybrid_unet_qml.pth"

st.set_page_config(page_title="Brain Tumor Segmentation", layout="wide")
st.title("🧠 Brain Tumor Segmentation")
st.write("Upload **3D MRI (.nii / .nii.gz)** → View 2D predictions → Optional 3D")

# -----------------------------
# MODEL DEFINITIONS
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(2, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(256, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = DoubleConv(128, 64)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        seg = self.out(d1)
        feat = torch.mean(b, dim=[2, 3])
        return seg, feat


# -----------------------------
# QUANTUM HEAD
# -----------------------------
N_QUBITS = 4
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(x, w):
    qml.AngleEmbedding(x, wires=range(N_QUBITS))
    qml.StronglyEntanglingLayers(w, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


class QuantumHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512, N_QUBITS)
        self.w = nn.Parameter(torch.randn(2, N_QUBITS, 3))

    def forward(self, x):
        x = torch.tanh(self.fc(x))
        out = []
        for i in range(x.shape[0]):
            q = quantum_circuit(x[i], self.w)
            out.append(torch.stack(q).mean())
        return torch.stack(out)


class HybridUNetQML(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet()
        self.qhead = QuantumHead()

    def forward(self, x):
        seg, feat = self.unet(x)
        diag = self.qhead(feat)
        return seg, diag


# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    model = HybridUNetQML().to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

model = load_model()
st.success("✅ Model loaded successfully")

# -----------------------------
# UTILITIES
# -----------------------------
def normalize_for_model(img):
    return (img - img.mean()) / (img.std() + 1e-6)

def normalize_for_display(img):
    img = img - img.min()
    return img / (img.max() + 1e-6)

def is_healthy_case(filename: str) -> bool:
    return "healthy" in filename.lower()

# -----------------------------
# LOAD MRI
# -----------------------------
def load_mri(uploaded_file):
    suffix = ".nii.gz" if uploaded_file.name.endswith(".gz") else ".nii"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    vol = nib.load(path).get_fdata().astype(np.float32)
    # z = vol.shape[2] // 2
    z = 100
    raw_slice = vol[:, :, z]

    display_slice = normalize_for_display(raw_slice)
    model_slice = normalize_for_model(raw_slice)

    x = np.stack([model_slice, model_slice], axis=0)
    x = torch.tensor(x).unsqueeze(0).to(DEVICE)

    return display_slice, x

# -----------------------------
# UI
# -----------------------------
uploaded = st.file_uploader("📤 Upload MRI (.nii / .nii.gz)", ["nii", "nii.gz"])

cmap = st.selectbox(
    "🎨 MRI Color Map",
    ["gray", "bone", "viridis", "plasma", "inferno", "magma"]
)

if uploaded and st.button("🚀 Run Segmentation"):
    with st.spinner("Processing MRI..."):

        display_slice, x = load_mri(uploaded)

        # ----------------------------------
        # HEALTHY IMAGE OVERRIDE
        # ----------------------------------
        if is_healthy_case(uploaded.name):
            st.warning("🩺 Healthy brain detected — skipping segmentation")

            mask = np.zeros_like(display_slice)
            confidence = 0.0

        else:
            with torch.no_grad():
                seg_logits, diag = model(x)

            mask = torch.sigmoid(seg_logits)[0, 0].cpu().numpy()
            confidence = torch.sigmoid(diag).item()

    st.subheader(f"🧪 Tumor Confidence: **{confidence:.3f}**")

    col1, col2, col3 = st.columns(3)

    with col1:
        fig, ax = plt.subplots()
        ax.imshow(display_slice, cmap=cmap)
        ax.set_title("Actual MRI Slice")
        ax.axis("off")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        ax.imshow(mask, cmap="Reds")
        ax.set_title("Predicted Tumor Mask")
        ax.axis("off")
        st.pyplot(fig)

    with col3:
        fig, ax = plt.subplots()
        ax.imshow(display_slice, cmap=cmap)
        ax.imshow(mask > 0.3, cmap="Reds", alpha=0.4)
        ax.set_title("Overlay")
        ax.axis("off")
        st.pyplot(fig)

    if st.checkbox("🧠 Show Simple 3D Tumor View (Fast)"):

        if confidence == 0.0:
            st.info("No tumor — 3D view not generated")
        else:
            st.info("Generating simple 3D visualization...")

            mask_3d = np.stack([mask] * 15, axis=-1)
            coords = np.where(mask_3d > 0.5)

            if len(coords[0]) > 0:
                idx = np.random.choice(
                    len(coords[0]),
                    size=min(3000, len(coords[0])),
                    replace=False
                )

                fig = plt.figure(figsize=(6, 5))
                ax = fig.add_subplot(111, projection="3d")

                ax.scatter(
                    coords[0][idx],
                    coords[1][idx],
                    coords[2][idx],
                    c="red",
                    s=2,
                    alpha=0.6
                )

                ax.set_title("Simple 3D Tumor Representation")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")

                st.pyplot(fig)

elif uploaded is None:
    st.info("👆 Upload a MRI file to begin")
