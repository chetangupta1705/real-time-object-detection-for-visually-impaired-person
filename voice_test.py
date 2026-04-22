import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

engine.say("Voice test successful. I am working properly.")
engine.runAndWait()
