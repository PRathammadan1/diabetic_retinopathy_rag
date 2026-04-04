import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import os

# we import a rag file first
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA

import base64

# -------- PAGE CONFIG --------

st.set_page_config(page_title="Diabetic Retinopathy Detection", layout="centered")

import time

title = " AI Powered Diabetic Retinopathy Detection"
placeholder = st.empty()

typed_text = ""

for char in title:
    typed_text += char
    placeholder.markdown(f"""
    <h1 style='
        text-align:center;
        color:white;
        text-shadow: 0 0 10px #ff4b2b, 0 0 20px #ff0000;
    '>{typed_text}</h1>
    """, unsafe_allow_html=True)
    time.sleep(0.03)

st.markdown("""
### My classes of diabetic is
- NO DR(there is no diabetic)
- Mild  (Early signs of diabetic retinopathy)
- Moderate (Moderate damage)
- Proliferate_DR (there is a chance of boarder line  diabetes)
- Severe (there is a very dangerous condition caused by diabetic)
""")
st.write("Upload a retina image and the AI model will predict the DR stage.")
# st.markdown("""
# <style>

# /* background gradient */
# .stApp {
#     background: linear-gradient(135deg,#e3f2fd,#ffffff);
# }

# /* title */
# .title{
#     font-size:42px;
#     font-weight:700;
#     text-align:center;
#     color:#0b5394;
# }

# /* subtitle */
# .subtitle{
#     text-align:center;
#     font-size:18px;
#     color:#555;
#     margin-bottom:30px;
# }

# /* cards */
# .card{
#     background:white;
#     padding:25px;
#     border-radius:15px;
#     box-shadow:0 4px 15px rgba(0,0,0,0.1);
# }

# /* prediction box */
# .result{
#     font-size:24px;
#     font-weight:bold;
#     text-align:center;
#     padding:15px;
#     border-radius:10px;
# }

# </style>
# """, unsafe_allow_html=True)
def get_base64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

current_dir = os.path.dirname(__file__)
img_path = os.path.join(current_dir, "retina_im.png")

img = get_base64(img_path)

st.markdown(f"""
<style>
.stApp {{
    background-image: url("data:image/png;base64,{img}");
    background-size: 110%;
    background-position: center;
    background-repeat: no-repeat;
    animation: zoomInOut 12s ease-in-out infinite alternate;

}}
/* SMOOTH ZOOM ANIMATION */
@keyframes zoomInOut {{
    0% {{
        background-size: 100%;
    }}
    50% {{
        background-size: 115%;
    }}
    100% {{
        background-size: 130%;
    }}
}}

/* OVERLAY (readability ke liye) */
.stApp::before {{
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(255,255,255,0.6);
    z-index: -1;
}}
</style>
""", unsafe_allow_html=True)

# -------- MODEL LOAD --------

model_path = os.path.join(os.path.dirname(__file__), "..", "best_resnet_model.pth")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 5)

model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

#----------------RAG------------
# Embeddings load
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Load FAISS index (jo tumne abhi banaya)
db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever()

llm = ChatOllama(
    model="phi",
    temperature=0
)

qa = RetrievalQA.from_chain_type(llm, retriever=retriever)

# -------- LABELS --------

labels = {
0: "No DR",
1: "Mild",
2: "Moderate",
3: "Severe",
4: "Proliferative DR"
}

# -------- IMAGE UPLOAD --------

uploaded_file = st.file_uploader("Upload Retina Image", type=["jpg","jpeg","png"])

# -------- TRANSFORM --------

transform = transforms.Compose([
transforms.Resize((224,224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# -------- MAIN LOGIC --------

if uploaded_file is not None:

 image = Image.open(uploaded_file).convert("RGB")
 image = image.resize((224,224))
 st.image(image, caption="Uploaded Retina Image")

 img_tensor = transform(image).unsqueeze(0)

 with torch.no_grad():
    output = model(img_tensor)
    pred_class = torch.argmax(output, dim=1).item()
    prediction_label = labels[pred_class]

 prob = torch.softmax(output, dim=1)
 confidence = prob[0][pred_class].item()

 st.success(f"Prediction: {labels[pred_class]}")
 st.write("Confidence:", round(confidence*100,2), "%")
 

#  if st.button("Generate Explanation"):
#     explanation = qa.run(f"Explain {prediction_label} diabetic retinopathy")
#     st.info("RAG Explanation:")
#     st.write(explanation)
st.markdown("Ask Questions About Your Result")

if "messages" not in st.session_state:
    st.session_state.messages = []

if uploaded_file is not None:
    if len(st.session_state.messages) == 0:
        try:
            explanation = qa.run(f"Explain {prediction_label} diabetic retinopathy in simple terms")
        except:
            explanation = "Explanation not available 😅"

        st.session_state.messages.append({
            "role": "assistant",
            "content": explanation
        })

st.markdown("## 🤖 AI Chatbot")

# messages show
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.write(f"🧑: {msg['content']}")
    else:
        st.write(f"🤖: {msg['content']}")

# input
user_input = st.text_input("Ask something...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    try:
        answer = qa.run(user_input)
    except:
        answer = "AI not working 😅"

    st.session_state.messages.append({"role": "assistant", "content": answer})