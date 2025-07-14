import torch
import torch.nn.functional as F
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from src.multimodal import MultimodalSentimentClassifier

# 1. Transcription CTC
def transcribe(audio_path: str) -> str:
    processor = Wav2Vec2Processor.from_pretrained(
        "jonatasgrosman/wav2vec2-large-xlsr-53-french",
        #cache_dir="./models"
    )
    model_ctc = Wav2Vec2ForCTC.from_pretrained(
        "jonatasgrosman/wav2vec2-large-xlsr-53-french",
        #cache_dir="./models"
    )

    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    inputs = processor(
        waveform.squeeze().numpy(),
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    with torch.no_grad():
        logits = model_ctc(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription.lower()

# 2. Inférence multimodale
def infer(audio_path: str) -> dict:
    # a) transcrire l’audio
    text = transcribe(audio_path)

    # b) charger et exécuter le modèle multimodal
    model = MultimodalSentimentClassifier()
    logits = model(audio_path, text)      # [1, n_classes]
    probs  = F.softmax(logits, dim=1).squeeze().tolist()

    labels = ["négatif", "neutre", "positif"]
    return { labels[i]: round(probs[i], 3) for i in range(len(labels)) }

# Test rapide en ligne de commande
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python src/inference.py <chemin_vers_audio.wav>")
        sys.exit(1)
    res = infer(sys.argv[1])
    print(f"Résultat multimodal : {res}")
