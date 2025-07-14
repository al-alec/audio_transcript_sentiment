import os
import re
from datetime import datetime

import gradio as gr
import torch
import pandas as pd
import soundfile as sf
import torchaudio

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from src.transcription import SpeechEncoder
from src.sentiment import TextEncoder

# Préchargement des modèles
processor_ctc = Wav2Vec2Processor.from_pretrained(
    "jonatasgrosman/wav2vec2-large-xlsr-53-french", cache_dir="./models"
    #"alec228/audio-sentiment/tree/main/wav2vec2", cache_dir="./models"
)
model_ctc = Wav2Vec2ForCTC.from_pretrained(
    "jonatasgrosman/wav2vec2-large-xlsr-53-french", cache_dir="./models"
    #"alec228/audio-sentiment/tree/main/wav2vec2", cache_dir="./models"
)

speech_enc = SpeechEncoder()
text_enc = TextEncoder()

# Pipeline d’analyse

def analyze_audio(audio_path):
    # Lecture et prétraitement
    data, sr = sf.read(audio_path)
    arr = data.T if data.ndim > 1 else data
    wav = torch.from_numpy(arr).unsqueeze(0).float()
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        sr = 16000
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Transcription
    inputs = processor_ctc(wav.squeeze().numpy(), sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        logits = model_ctc(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)
    transcription = processor_ctc.batch_decode(pred_ids)[0].lower()

    # Sentiment principal
    sent_dict = TextEncoder.analyze_sentiment(transcription)
    label, conf = max(sent_dict.items(), key=lambda x: x[1])
    emojis = {"positif": "😊", "neutre": "😐", "négatif": "☹️"}
    emoji = emojis.get(label, "")

    # Segmentation par phrase
    segments = [s.strip() for s in re.split(r'[.?!]', transcription) if s.strip()]
    seg_results = []
    for seg in segments:
        sd = TextEncoder.analyze_sentiment(seg)
        l, c = max(sd.items(), key=lambda x: x[1])
        seg_results.append({"Segment": seg, "Sentiment": l.capitalize(), "Confiance (%)": round(c*100,1)})
    seg_df = pd.DataFrame(seg_results)

    # Historique entry
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history_entry = {
        "Horodatage": timestamp,
        "Transcription": transcription,
        "Sentiment": label.capitalize(),
        "Confiance (%)": round(conf*100,1)
    }

    # Rendu
    summary_html = (
        f"<div style='display:flex;align-items:center;'>"
        f"<span style='font-size:3rem;margin-right:10px;'>{emoji}</span>"
        f"<h2 style='color:#6a0dad;'>{label.upper()}</h2>"
        f"</div>"
        f"<p><strong>Confiance :</strong> {conf*100:.1f}%</p>"
    )
    return transcription, summary_html, seg_df, history_entry

# Export CSV

def export_history_csv(history):
    df = pd.DataFrame(history)
    path = "history.csv"
    df.to_csv(path, index=False)
    return path

# Interface Chat + historique

demo = gr.Blocks(theme=gr.themes.Monochrome(primary_hue="purple"))
with demo:
    gr.Markdown("# Chat & Analyse de Sentiment Audio")

    gr.HTML("""
    <div style="display: flex; flex-direction: column; gap: 10px; margin-bottom: 20px;">
        <div style="background-color: #f3e8ff; padding: 12px 20px; border-radius: 12px; border-left: 5px solid #8e44ad;">
            <strong>Étape 1 :</strong> Enregistrez votre voix ou téléversez un fichier audio (format WAV recommandé).
        </div>
        <div style="background-color: #e0f7fa; padding: 12px 20px; border-radius: 12px; border-left: 5px solid #0097a7;">
            <strong>Étape 2 :</strong> Cliquez sur le bouton <em><b>Analyser</b></em> pour lancer la transcription et l’analyse.
        </div>
        <div style="background-color: #fff3e0; padding: 12px 20px; border-radius: 12px; border-left: 5px solid #fb8c00;">
            <strong>Étape 3 :</strong> Visualisez les résultats : transcription, sentiment, et analyse détaillée.
        </div>
        <div style="background-color: #e8f5e9; padding: 12px 20px; border-radius: 12px; border-left: 5px solid #43a047;">
            <strong>Étape 4 :</strong> Exportez l’historique des analyses au format CSV si besoin.
        </div>
    </div>

    <script>
        const origin = window.location.origin;
        const swaggerUrl = origin + "/docs";
        document.getElementById("swagger-link").innerHTML = `<a href="${swaggerUrl}" target="_blank">Voir la documentation de l’API (Swagger)</a>`;
    </script>
    """)

    with gr.Row():
        with gr.Column(scale=2):
            audio_in = gr.Audio(sources=["microphone","upload"], type="filepath", label="Audio Input")
            btn = gr.Button("Analyser")
            export_btn = gr.Button("Exporter CSV")
        with gr.Column(scale=3):
            chat = gr.Chatbot(label="Historique des échanges")
            transcription_out = gr.Textbox(label="Transcription", interactive=False)
            summary_out = gr.HTML(label="Sentiment")
            seg_out = gr.Dataframe(label="Détail par segment")
            hist_out = gr.Dataframe(label="Historique")

    state_chat = gr.State([])  # list of (user,bot)
    state_hist = gr.State([])  # list of dict entries

    def chat_callback(audio_path, chat_history, hist_state):
        transcription, summary, seg_df, hist_entry = analyze_audio(audio_path)
        user_msg = "[Audio reçu]"
        bot_msg = f"**Transcription :** {transcription}\n**Sentiment :** {summary}"
        chat_history = chat_history + [(user_msg, bot_msg)]
        hist_state = hist_state + [hist_entry]
        return chat_history, transcription, summary, seg_df, hist_state

    btn.click(
        fn=chat_callback,
        inputs=[audio_in, state_chat, state_hist],
        outputs=[chat, transcription_out, summary_out, seg_out, state_hist]
    )
    export_btn.click(
        fn=export_history_csv,
        inputs=[state_hist],
        outputs=[gr.File(label="Télécharger CSV")]
    )



if __name__ == "__main__":
    demo.launch()