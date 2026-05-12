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
st.markdown(
    "<p style='font-weight:bold; color:#856404; background-color:#fff3cd; padding:10px; border-radius:8px;'>⚠️ This is an AI-based assistive tool. Not a medical diagnosis.</p>",
    unsafe_allow_html=True
)

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
<div style='
    background: rgba(255,255,255,0.8);
    padding:20px;
    border-radius:15px;
    box-shadow:0 4px 15px rgba(0,0,0,0.2);
'>

<h3 style='text-align:center; color:#0b5394;'> Diabetic Retinopathy Classes</h3>

<ul style='font-size:17px; color:#333;'>

<li><b style='color:green;'>No DR</b> → No diabetic condition</li>

<li><b style='color:#ffc107;'>Mild</b> → Early signs detected</li>

<li><b style='color:#fd7e14;'>Moderate</b> → Moderate damage</li>

<li><b style='color:#dc3545;'>Severe</b> → High risk condition</li>

<li><b style='color:#6f42c1;'>Proliferative DR</b> → Advanced stage</li>

</ul>

</div>
""", unsafe_allow_html=True)

st.write(" ")

st.markdown("""
<div style='
    background-color:#e3f2fd;
    padding:15px;
    border-radius:10px;
    text-align:center;
    font-size:18px;
    color:#0d47a1;
    font-weight:500;
'>
Upload a retinal image and let the AI predict the stage of Diabetic Retinopathy
</div>
""", unsafe_allow_html=True)


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

st.markdown("""
<div style='
    background-color:#f1f8ff;
    padding:20px;
    border-radius:12px;
    text-align:center;
'>
    <h4> Upload Retina Image</h4>
    <p style='color:gray;'>Supported formats: JPG, JPEG, PNG</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)

    #  resize yaha
    image = image.resize((202, 204))

    st.image(image, caption="Uploaded Retina Image")

# -------- TRANSFORM --------

transform = transforms.Compose([
transforms.Resize((224,224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# -------- MAIN LOGIC --------
if st.button("Analyze Image"):
 if uploaded_file is not None:

  image = Image.open(uploaded_file).convert("RGB")
  image = image.resize((224,224))
  st.image(image, caption="Uploaded Retina Image")

  img_tensor = transform(image).unsqueeze(0)

  with st.spinner("🔍 Analyzing retina image..."):

    with torch.no_grad():
      output = model(img_tensor)
      pred_class = torch.argmax(output, dim=1).item()
      prediction_label = labels[pred_class]

    prob = torch.softmax(output, dim=1)
    confidence = prob[0][pred_class].item()
  st.session_state.prediction_label = prediction_label
  st.session_state.confidence = confidence

  st.markdown(
    f"""
    <div style='background-color:#d4edda; padding:15px; border-radius:10px; text-align:center;'>
        <p style='font-size:35px; font-weight:bold; color:#155724;'>
            Prediction: {labels[pred_class]}
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

  st.markdown(
    f"""
    <p style='text-align:center; font-size:22px; font-weight:bold; color:#0c5460;'>
        Confidence: {round(confidence*100,2)}%
    </p>
    """,
    unsafe_allow_html=True
)
 

#  if st.button("Generate Explanation"):
#     explanation = qa.run(f"Explain {prediction_label} diabetic retinopathy")
#     st.info("RAG Explanation:")
#     st.write(explanation)
st.markdown("Ask Questions About Your Result")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "prediction_label" in st.session_state:
    if len(st.session_state.messages) == 0:
        try:
            with st.spinner(" Generating explanation..."):
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

import streamlit as st
import requests

st.title("Geocoding App")

address = st.text_input("Enter address")

if st.button("Get Location"):
    res = requests.get(
        "http://127.0.0.1:8000/geocode",
        params={"address": address}
    )

    data = res.json()
    st.json(data)
if st.button("Find Places"):
    res = requests.get("http://127.0.0.1:8000/places")
    st.json(res.json())
