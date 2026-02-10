# ============================================
# DETECÇÃO DE PNEUMONIA + GRAD-CAM (AUTograd - MAIS CONFIÁVEL)
# ============================================
# Esta versão usa torch.autograd em vez de hooks, que é mais confiável

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import gradio as gr
import cv2
import numpy as np
import traceback

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Dispositivo: {device}")

# --------------------------------------------
# 1. Dataset
# --------------------------------------------
def create_xray_image(label, seed=42):
    np.random.seed(seed)
    img = np.ones((224, 224), dtype=np.uint8) * 220
    if label == 1:
        for _ in range(3):
            x, y = np.random.randint(50, 170), np.random.randint(50, 170)
            r = np.random.randint(15, 30)
            cv2.circle(img, (x, y), r, 255, -1)
    noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    return Image.fromarray(img).convert('RGB')

class SyntheticXRayDataset(Dataset):
    def __init__(self, num_samples=20, transform=None):
        self.transform = transform
        self.images, self.labels = [], []
        for i in range(num_samples):
            self.labels.append(i % 2)
            self.images.append(create_xray_image(i % 2, seed=i))
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img = self.transform(self.images[idx]) if self.transform else self.images[idx]
        return img, self.labels[idx]

transform = transforms.Compose([
    transforms.Resize(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_loader = DataLoader(SyntheticXRayDataset(40, transform), batch_size=8, shuffle=True)

# --------------------------------------------
# 2. Modelo
# --------------------------------------------
try:
    from torchvision.models import ResNet18_Weights
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
except:
    model = models.resnet18(pretrained=True)

for p in model.parameters():
    p.requires_grad = False
for n, p in model.named_parameters():
    if 'layer4' in n or 'fc' in n:
        p.requires_grad = True

model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# Treino
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
for epoch in range(3):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        nn.CrossEntropyLoss()(model(inputs), labels).backward()
        optimizer.step()
print("✅ Modelo treinado!")

# --------------------------------------------
# 3. GRAD-CAM COM AUTograd (SEM HOOKS)
# --------------------------------------------
class GradCAMAutograd:
    """
    Grad-CAM usando torch.autograd.grad - mais confiável que hooks
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

    def __call__(self, input_tensor, target_class=None):
        """
        Gera Grad-CAM

        Args:
            input_tensor: Tensor da imagem (1, C, H, W)
            target_class: Classe alvo (None = predição)

        Returns:
            heatmap: Array numpy
            pred_class: Classe predita
        """
        # Habilitar gradientes
        input_tensor = input_tensor.clone().detach().requires_grad_(True)

        # Ativar modo de gradiente
        with torch.enable_grad():
            # Forward pass mantendo ativações
            activations = None

            def hook_fn(module, input, output):
                nonlocal activations
                activations = output

            handle = self.target_layer.register_forward_hook(hook_fn)

            # Forward
            self.model.eval()
            output = self.model(input_tensor)

            # Remover hook
            handle.remove()

            if activations is None:
                raise RuntimeError("Não foi possível capturar ativações")

            # Determinar classe
            if target_class is None:
                target_class = output.argmax(dim=1).item()

            # Calcular gradientes usando autograd.grad
            score = output[0, target_class]

            # Obter gradientes da ativação em relação ao score
            grads = torch.autograd.grad(
                outputs=score,
                inputs=activations,
                retain_graph=False,
                create_graph=False
            )[0]

        # Grad-CAM: pesos = média global dos gradientes
        weights = grads.mean(dim=[2, 3], keepdim=True)

        # Combinar com ativações
        cam = (weights * activations).sum(dim=1)

        # ReLU e normalização
        cam = torch.relu(cam)
        cam = cam.squeeze()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.detach().cpu().numpy(), target_class

# Criar Grad-CAM
target_conv = model.layer4[1].conv2
gradcam = GradCAMAutograd(model, target_conv)
print(f"✅ Grad-CAM pronto - camada: {target_conv}")

# --------------------------------------------
# 4. Funções
# --------------------------------------------
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image):
    if image is None:
        return "⚠️ Envie uma imagem!"

    img = image.convert('RGB')
    img_t = inference_transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        probs = torch.nn.functional.softmax(model(img_t)[0], dim=0)

    pneumonia = float(probs[1]) * 100
    result = "⚠️ PNEUMONIA" if pneumonia > 50 else "✅ NORMAL"
    return f"{result}\n\nPNEUMONIA: {pneumonia:.1f}%"

def gradcam_inference(image):
    if image is None:
        return None, "⚠️ Envie uma imagem!"

    try:
        img_rgb = image.convert('RGB')
        original_np = np.array(img_rgb)

        img_t = inference_transform(img_rgb).unsqueeze(0).to(device)

        # Gerar Grad-CAM
        heatmap, pred = gradcam(img_t)

        # Visualização
        h, w = original_np.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))

        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(original_np, 0.5, heatmap_color, 0.5, 0)

        # Triplet
        size = (300, 300)
        orig = img_rgb.resize(size)
        hm = Image.fromarray(cv2.resize(heatmap_color, size))
        ov = Image.fromarray(cv2.resize(overlay, size))

        triplet = Image.new('RGB', (900, 300))
        triplet.paste(orig, (0, 0))
        triplet.paste(hm, (300, 0))
        triplet.paste(ov, (600, 0))

        msg = "🩺 PNEUMONIA" if pred == 1 else "✅ NORMAL"
        return triplet, msg

    except Exception as e:
        err = traceback.format_exc()
        print(err)
        return None, f"❌ Erro: {str(e)}"

# --------------------------------------------
# 5. Interface
# --------------------------------------------
with gr.Blocks(title="🩺 Pneumonia + Grad-CAM") as demo:
    gr.Markdown("# 🩺 Detector de Pneumonia com Grad-CAM")
    gr.Markdown("### Usando torch.autograd (mais confiável)")

    with gr.Tabs():
        with gr.TabItem("🩻 Classificação"):
            img1 = gr.Image(type="pil", label="Raio-X")
            btn1 = gr.Button("Analisar", variant="primary")
            out1 = gr.Textbox(label="Resultado", lines=3)
            btn1.click(predict, inputs=img1, outputs=out1)

        with gr.TabItem("🔬 Grad-CAM"):
            img2 = gr.Image(type="pil", label="Raio-X")
            btn2 = gr.Button("Gerar Heatmap 🔥", variant="primary")
            img_out = gr.Image(label="Original | Heatmap | Overlay")
            txt_out = gr.Textbox(label="Status", lines=2)
            btn2.click(gradcam_inference, inputs=img2, outputs=[img_out, txt_out])

    gr.Markdown("⚠️ Demonstração educacional")

demo.launch(share=True)
