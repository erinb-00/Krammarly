import joblib
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import re
from nltk.tokenize import sent_tokenize
from statistics import mode

# Load the classifier
lr_clf = joblib.load('../Model/domain_classifier.pkl')

# Load BERT tokenizer and model (assuming they're saved in these folders)
tokenizer = AutoTokenizer.from_pretrained('../Model/klue_bert_tokenizer')
model = AutoModel.from_pretrained('../Model/klue_bert_model')


# Pre-loaded components (make sure you've already run these)
# lr_clf = joblib.load('domain_classifier.pkl')
# tokenizer = AutoTokenizer.from_pretrained('./klue_bert_tokenizer')
# model = AutoModel.from_pretrained('./klue_bert_model')

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

def classify_input(text):
    # DOMAIN: Tokenize and get CLS embedding
    input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)[:512]])
    attention_mask = (input_ids != 0).long()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    cls_embedding = outputs[0][:, 0, :].numpy()
    prediction = lr_clf.predict(cls_embedding)[0]
    labels = ['Academic', 'General', 'Colloquial', 'Literary']
    domain_label = labels[prediction]

    # FORMALITY: Run sentence-wise honorific check
    sentences = sent_tokenize(text)
    styles = [honorifics(s) for s in sentences]
    formality = mode(styles) if styles else "Unknown"

    return {'Domain': domain_label, 'Formality': formality}

sample1 = "유아교사의 핵심역량 8개 영역은 교사인성 및 전문성 개발, 놀이지원, 유아의 성장과 발달 등으로 구성되었다."
print("Sample Text:", sample1)
print("Predicted Domain:", classify_input(sample1))
print("\n")

sample2 = "안녕하세요? 오늘은 유아교육의 중요성에 대해 이야기해볼게요!"
result = classify_input(sample2)
print("Sample Text:", sample2)
print("Predicted Domain:", result['Domain'])
print("Formality:", result['Formality'])