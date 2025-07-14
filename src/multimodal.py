# src/multimodal.py

from .transcription import SpeechEncoder
from .sentiment     import TextEncoder
import torch
import torch.nn as nn

class MultimodalSentimentClassifier(nn.Module):
    def __init__(
        self,
        wav2vec_name: str = "jonatasgrosman/wav2vec2-large-xlsr-53-french",
        bert_name:    str = "nlptown/bert-base-multilingual-uncased-sentiment",
        cache_dir:    str = "./models",
        hidden_dim:   int = 256,
        n_classes:    int = 3
    ):
        super().__init__()
        self.speech_encoder = SpeechEncoder(
            model_name=wav2vec_name,
            cache_dir=cache_dir
        )
        self.text_encoder = TextEncoder(
            model_name=bert_name,
            cache_dir=cache_dir
        )
        dim_a = self.speech_encoder.model.config.hidden_size
        dim_t = self.text_encoder.model.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(dim_a + dim_t, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, audio_path: str, text: str) -> torch.Tensor:
        a_feat = self.speech_encoder.extract_features(audio_path)
        t_feat = self.text_encoder.extract_features([text])
        fused  = torch.cat([a_feat, t_feat], dim=1)
        return self.classifier(fused)
