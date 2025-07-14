from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import os

# 1) Export du modèle de transcription (Wav2Vec2)
wv_model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-french"
os.makedirs("hf_model/wav2vec2", exist_ok=True)
Wav2Vec2ForCTC.from_pretrained(wv_model_name).save_pretrained("hf_model/wav2vec2")
Wav2Vec2Processor.from_pretrained(wv_model_name).save_pretrained("hf_model/wav2vec2")

# 2) Export du modèle de sentiment (BERT)
sent_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
os.makedirs("hf_model/bert-sentiment", exist_ok=True)
AutoModelForSequenceClassification.from_pretrained(sent_model_name).save_pretrained("hf_model/bert-sentiment")
AutoTokenizer.from_pretrained(sent_model_name).save_pretrained("hf_model/bert-sentiment")
