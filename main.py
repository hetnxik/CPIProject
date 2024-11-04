import streamlit as st
import sounddevice as sd
import wavio
import tempfile
from gtts import gTTS
import speech_recognition as sr
import time

import model
from llm import get_mood_from_text
from model import spotify_df


def text_to_speech(text):
    tts = gTTS(text)
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp.name)
    return temp.name


st.title("Mood Analysis")

if "is_recording" not in st.session_state:
    st.session_state.is_recording = False
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None
if "recorded" not in st.session_state:
    st.session_state.recorded = False
if "start_time" not in st.session_state:
    st.session_state.start_time = None

st.subheader("Choose Input Method")
input_choice = st.radio("Select input type:", ("Text", "Audio"))


if input_choice == "Text":
    user_text = st.text_area("Enter text for mood analysis:")
    if st.button("Analyze Mood from Text"):
        mood = get_mood_from_text(user_text)
        st.write(f"Identified Mood: {mood}")
        rec = model.recommend_songs(mood=mood, spotify_df=spotify_df)
        song_list = rec.apply(lambda row: f"{row['name']} - {row['artist']}", axis=1).tolist()

        st.write(song_list)

        audio_file = text_to_speech(f"Recommended Songs: {song_list}")
        st.audio(audio_file)


else:
    fs = 44100  # Sample rate
    if st.button("Start Recording") and not st.session_state.is_recording:
        st.session_state.is_recording = True
        st.session_state.recorded = False
        st.session_state.start_time = time.time()
        st.session_state.recording = sd.rec(int(600 * fs), samplerate=fs, channels=2)
        st.write("Recording... Speak now.")

    if st.button("Stop Recording") and st.session_state.is_recording:
        sd.stop()
        st.session_state.is_recording = False
        end_time = time.time()
        duration = int(end_time - st.session_state.start_time) + 1

        trimmed_recording = st.session_state.recording[:duration * fs]

        temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        wavio.write(temp_audio_path.name, trimmed_recording, fs, sampwidth=2)
        st.session_state.audio_path = temp_audio_path.name
        st.session_state.recorded = True

    if st.session_state.recorded and st.session_state.audio_path:
        st.audio(st.session_state.audio_path, format="audio/wav")

        if st.button("Analyze Mood from Audio"):
            recognizer = sr.Recognizer()
            with sr.AudioFile(st.session_state.audio_path) as source:
                audio_data = recognizer.record(source)
                try:
                    user_input = recognizer.recognize_google(audio_data)
                    st.write(f"Transcribed Text: {user_input}")
                    mood = get_mood_from_text(user_input)
                    st.write(f"Identified Mood: {mood}")
                    rec = model.recommend_songs(mood=mood, spotify_df=spotify_df)
                    song_list = rec.apply(lambda row: f"{row['name']} - {row['artist']}", axis=1).tolist()

                    st.write(song_list)

                    audio_file = text_to_speech(f"Recommended Songs: {song_list}")
                    st.audio(audio_file)
                except sr.UnknownValueError:
                    st.write("Could not understand audio.")
                except sr.RequestError as e:
                    st.write(f"Speech recognition error: {e}")


