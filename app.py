import streamlit as st
import tempfile
import os
from pydub import AudioSegment
import io
from transformers import pipeline
import torch

## Functions
def audio_to_text(audio_input):
    """
    Convert audio to text using the Whisper model.
    """
    audio = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", return_timestamps=True)
    text = audio(audio_input)
    return text

def text_to_Speech(text_input):
    """
    Convert text to speech using the MMS TTS model.
    """
    text_to_speeches = pipeline("text-to-speech", model="facebook/mms-tts-eng")
    audio = text_to_speeches(text_input)
    return audio

def summary_text(text_input):
    """
    Summarize text using the BART model.
    """
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text_input)
    return summary

## Streamlit App
st.set_page_config(page_title="Audio Processing App", layout="wide")
st.title("Audio Transcription & Summarization")

# File upload section
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg", "flac"])

if uploaded_file is not None:
    # Display audio player
    st.subheader("Uploaded Audio")
    st.audio(uploaded_file)
    
    # Transcription section
    if st.button("Generate Transcript"):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            # Convert to WAV format using pydub
            audio = AudioSegment.from_file(uploaded_file)
            audio.export(tmp_file.name, format="wav")
            tmp_path = tmp_file.name
            
        result = audio_to_text(tmp_path)
        st.session_state.transcript = result["text"]
        os.unlink(tmp_path)
    
    if 'transcript' in st.session_state:
        st.subheader("Transcript")
        st.text_area("Transcript Text", st.session_state.transcript, height=200)
        st.download_button(
            label="Download Transcript",
            data=st.session_state.transcript,
            file_name="transcript.txt",
            mime="text/plain"
        )
        
        # Summarization section
        if st.button("Generate Summary"):
            summary = summary_text(st.session_state.transcript)
            st.session_state.summary_text = summary[0]['summary_text']
    
    if 'summary_text' in st.session_state:
        st.subheader("Summary")
        st.text_area("Summary Text", st.session_state.summary_text, height=100)
        st.download_button(
            label="Download Summary",
            data=st.session_state.summary_text,
            file_name="summary.txt",
            mime="text/plain"
        )
        
        # Text-to-speech section
        if st.button("Convert Summary to Audio"):
            audio_output = text_to_Speech(st.session_state.summary_text)
            st.session_state.audio_data = audio_output
    
    if 'audio_data' in st.session_state:
        st.subheader("Summary Audio")
        sampling_rate = st.session_state.audio_data['sampling_rate']
        audio_array = st.session_state.audio_data['audio']
        
        # Convert numpy array to audio bytes
        byte_io = io.BytesIO()
        write(byte_io, sampling_rate, audio_array)
        byte_io.seek(0)
        
        st.audio(byte_io, format='audio/wav')
        st.download_button(
            label="Download Summary Audio",
            data=byte_io,
            file_name="summary_audio.wav",
            mime="audio/wav"
        )
