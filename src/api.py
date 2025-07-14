import tempfile
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch.nn.functional as F
import torchaudio
import torch

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from src.transcription import SpeechEncoder
from src.sentiment     import TextEncoder
from src.multimodal    import MultimodalSentimentClassifier

app = FastAPI(
    title="API Multimodale de Transcription & Sentiment",
    version="1.0"
)

# Précharge des modèles
processor_ctc = Wav2Vec2Processor.from_pretrained(
    "jonatasgrosman/wav2vec2-large-xlsr-53-french",
    #"jonatasgrosman/wav2vec2-large-xlsr-53-french",
    cache_dir="./models"
)
model_ctc = Wav2Vec2ForCTC.from_pretrained(
    "jonatasgrosman/wav2vec2-large-xlsr-53-french",
    #"alec228/audio-sentiment/tree/main/wav2vec2",
    cache_dir="./models"
)
speech_enc = SpeechEncoder()
text_enc   = TextEncoder()
model_mm   = MultimodalSentimentClassifier()

def transcribe_ctc(wav_path: str) -> str:
    waveform, sr = torchaudio.load(wav_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    inputs = processor_ctc(
        waveform.squeeze().numpy(),
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    with torch.no_grad():
        logits = model_ctc(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return processor_ctc.batch_decode(pred_ids)[0].lower()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Vérifier le type
    if not file.filename.lower().endswith((".wav", ".flac", ".mp3")):
        raise HTTPException(status_code=400,
            detail="Seuls les fichiers audio WAV/FLAC/MP3 sont acceptés.")
    # 2. Sauvegarder temporairement
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # 3. Transcription
        transcription = transcribe_ctc(tmp_path)

        # 4. Features multimodales
        audio_feat = speech_enc.extract_features(tmp_path)
        text_feat  = text_enc.extract_features([transcription])

        # 5. Classification
        logits = model_mm.classifier(torch.cat([audio_feat, text_feat], dim=1))
        probs  = F.softmax(logits, dim=1).squeeze().tolist()
        labels = ["négatif", "neutre", "positif"]
        sentiment = { labels[i]: round(probs[i], 3) for i in range(len(labels)) }

        return JSONResponse({
            "transcription": transcription,
            "sentiment": sentiment
        })

    finally:
        # 6. Nettoyage
        os.remove(tmp_path)
