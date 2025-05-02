from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModel
import torch
import joblib
import numpy as np
import re
from nltk.tokenize import sent_tokenize
from statistics import mode

app = Flask(__name__)

# Load saved model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("../Model/klue_bert_tokenizer")
model = AutoModel.from_pretrained("../Model/klue_bert_model")
classifier = joblib.load("../Model/domain_classifier.pkl")

# Honorific classifier (rule-based)
def honorifics(input):
    reFormal = r"(니다[.!]|니까[.?]*)"
    groupFormal = re.search(reFormal, input)
    reSemi = r"(요[.!?]*)"
    groupSemi = re.search(reSemi, input)
    if groupFormal:
        return 'Formal Honorific'
    elif groupSemi:
        return 'Casual Honorific'
    else:
        return 'Non-honorific'

# Domain classifier (BERT + logistic regression)
def domain(text):
    input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)[:512]])
    with torch.no_grad():
        outputs = model(input_ids)
    cls_embedding = outputs[0][:, 0, :].numpy()
    prob = classifier.predict_proba(cls_embedding)
    index = np.argmax(prob)
    genres = ['Academic', 'General', 'Colloquial', 'Literary']
    return genres[index]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        text = request.form['text']
        dom = domain(text)
        honor = [honorifics(s) for s in sent_tokenize(text)]
        audience = mode(honor) if honor else "N/A"
        prediction = f"Domain: {dom} | Formality: {audience}"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
