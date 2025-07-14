# src/transcription.py

from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
import torchaudio

class SpeechEncoder:
    def __init__(
        self,
        model_name: str = "jonatasgrosman/wav2vec2-large-xlsr-53-french",
        #model_name: str = "alec228/audio-sentiment/tree/main/wav2vec2",
        cache_dir: str = "./models"
    ):
        # Processor pour prétraiter l'audio
        self.processor = Wav2Vec2Processor.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        # Modèle de base (sans tête CTC)
        self.model = Wav2Vec2Model.from_pretrained(
            model_name, cache_dir=cache_dir
        )

    def extract_features(self, audio_path: str) -> torch.Tensor:
        """
        Charge un fichier audio, le resample à 16 kHz, convertit en mono,
        et renvoie la représentation vectorielle moyenne sur la séquence.
        """
        # 1. Chargement
        waveform, sample_rate = torchaudio.load(audio_path)

        # 2. Resample si nécessaire
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=16000
            )(waveform)

        # 3. Passage en mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # 4. Prétraitement pour le modèle
        inputs = self.processor(
            waveform.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        # 5. Extraction sans gradient
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 6. Moyenne temporelle des embeddings
        return outputs.last_hidden_state.mean(dim=1)  # shape: [batch, hidden_size]
