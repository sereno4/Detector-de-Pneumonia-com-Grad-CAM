# Detector-de-Pneumonia-com-Grad-CAM
Sistema treinado para detectar pneumonia através de fotos

# 🩺 Detector de Pneumonia com Grad-CAM

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white&style=for-the-badge)](https://pytorch.org)
[![Gradio](https://img.shields.io/badge/Gradio-F472B6?logo=gradio&logoColor=white&style=for-the-badge)](https://gradio.app)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD166?logo=huggingface&logoColor=black&style=for-the-badge)](https://huggingface.co)

> 🔥 **Detector de pneumonia em raio-X com Grad-CAM explicável usando `torch.autograd.grad`**

[![Demo](https://i.imgur.com/placeholder-pneumonia.png)](https://huggingface.co/spaces/Danielfonseca1212/pneumonia-detector)

🔗 **Experimente online:** https://huggingface.co/spaces/Danielfonseca1212/pneumonia-detector  
📂 **Repositório:** https://github.com/Danielfonseca1212/pneumonia-detector

---

## 🎯 Tecnologias Utilizadas

| Tecnologia | Papel no Projeto |
|------------|------------------|
| **PyTorch 2.0+** | Framework principal com `torch.autograd.grad` robusto |
| **Grad-CAM sem hooks** | Evita erros com camadas congeladas (técnica avançada) |
| **OpenCV** | Processamento de imagens e overlays coloridos |
| **Gradio** | Interface web interativa com abas |
| **Hugging Face Spaces** | Deploy em nuvem com 1 clique |

---

## 💡 Por Que Este Projeto se Destaca?

✅ **Grad-CAM com `torch.autograd.grad`** — mais confiável que hooks tradicionais  
✅ **Funciona mesmo com camadas congeladas** — solução para erro sistemático comum  
✅ **Visualização tripla** — Original \| Heatmap \| Overlay  
✅ **Aplicação médica real** — detecção de pneumonia em raio-X  

> 📊 **Recrutadores veem centenas de classificadores básicos. O que impressiona é interpretabilidade aplicada a domínios críticos como saúde.**

---

## 🚀 Como Rodar Localmente

```bash
git clone https://github.com/Danielfonseca1212/pneumonia-detector.git
cd pneumonia-detector

📁 Estrutura do Projeto

pneumonia-detector/
├── app.py              # Interface Gradio com Grad-CAM
├── requirements.txt    # Dependências compatíveis
└── README.md           # Documentação profissional

🔗 Links Diretos
Plataforma
Link
App Online
https://huggingface.co/spaces/Danielfonseca1212/pneumonia-detector
Hugging Face
https://huggingface.co/spaces/Danielfonseca1212/pneumonia-detector
GitHub
https://github.com/Danielfonseca1212/pneumonia-detector


pip install -r requirements.txt
python app.py
