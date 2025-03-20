# import datetime
# import pyttsx3
# import speech_recognition as sr
# import subprocess
# import streamlit as st
# engine = pyttsx3.init()


# voices = engine.getProperty('voices') 
# engine.setProperty('voice', voices[1].id)

# engine.setProperty('volume',1.0)

# def speak(audio):
#     engine.say(audio)
#     print(audio)
#     engine.runAndWait()

# def takecommand():
#     r = sr.Recognizer()
#     with sr.Microphone() as source:
#         print('Listening...')
#         r.pause_threshold = 1
#         r.adjust_for_ambient_noise(source, duration=1)  # Adjust for ambient noise
#         try:
#             audio = r.listen(source, timeout=5)  # Increase timeout if needed
#             print('Recognizing...')
#             query = r.recognize_google(audio, language='en-in')
#             print(f"User said: {query}\n")
#             return query
#         except sr.UnknownValueError:
#             print("Sorry, I didn't catch that. Could you please repeat?")
#             speak("Sorry, I didn't catch that. Could you please repeat?")
#             audio = r.listen(source, timeout=10)  # Increase timeout if needed
#             print('Recognizing...')
#             query = r.recognize_google(audio, language='en-in')
#             print(f"User said: {query}\n")
#             return query
#         except sr.RequestError:
#             print("Sorry, there was an issue with recognizing your voice.")
#             return "None"
    
# def userName():
#     speak("What should i call you?")
#     uname = takecommand()
#     speak("Welcome "+ uname)
#     speak("How Can i help you?")
    
# def wishMe():
#     hour = int(datetime.datetime.now().hour)
#     if hour >= 0 and hour<12:
#         speak('Good Morning!')
#     elif hour>=12 and hour<18:
#         speak('Good Afternoon!')
#     else:
#         speak("Good Evening !")
#     speak("i am your virtual assitance ")
    
# def openProjectLink():
#     # Implement the function to open the NER streamlit project or perform any other action
    

#     subprocess.Popen(['streamlit', 'run', 'invoice_predict.py'])
#     speak("Opening the NER streamlit project.")  

# if __name__ == '__main__':
#     wishMe()
#     userName()
#     while True:
#         order= takecommand().lower()
#         if 'open project' in order:
#             openProjectLink()
            
#         elif 'exit' in order:
#             speak("Goodbye!")
#             break
    
    
    
import datetime
import pyttsx3
import speech_recognition as sr
import subprocess
import webbrowser
import streamlit as st

engine = pyttsx3.init()

# Set voice properties
voices = engine.getProperty('voices') 
engine.setProperty('voice', voices[1].id)
engine.setProperty('volume', 1.0)

def speak(audio):
    engine.say(audio)
    print(audio)
    engine.runAndWait()

def takecommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print('Listening...')
        r.pause_threshold = 1
        r.adjust_for_ambient_noise(source, duration=1)  # Adjust for ambient noise
        try:
            audio = r.listen(source, timeout=5)  # Increase timeout if needed
            print('Recognizing...')
            query = r.recognize_google(audio, language='en-in')
            print(f"User said: {query}\n")
            return query
        except sr.UnknownValueError:
            print("Sorry, I didn't catch that. Could you please repeat?")
            speak("Sorry, I didn't catch that. Could you please repeat?")
            audio = r.listen(source, timeout=10)  # Increase timeout if needed
            print('Recognizing...')
            query = r.recognize_google(audio, language='en-in')
            print(f"User said: {query}\n")
            return query
        except sr.RequestError:
            print("Sorry, there was an issue with recognizing your voice.")
            return "None"
    
def userName():
    speak("What should I call you?")
    uname = takecommand()
    speak("Welcome " + uname)
    speak("How can I help you?")
    
def wishMe():
    hour = int(datetime.datetime.now().hour)
    if hour >= 0 and hour < 12:
        speak('Good Morning!')
    elif hour >= 12 and hour < 18:
        speak('Good Afternoon!')
    else:
        speak("Good Evening!")
    speak("I am your virtual assistant.")
    
def openProjectLink():
    subprocess.Popen(['streamlit', 'run', 'invoice_predict.py'])
    speak("Opening the NER Streamlit project.")

def openBrowser():
    speak("Opening the browser.")
    webbrowser.open('http://www.google.com')

def openYouTube():
    speak("Opening YouTube.")
    webbrowser.open('https://www.youtube.com')

def openApp(app_name):
    try:
        if app_name.lower() == "chrome":
            subprocess.Popen(["chrome"])  # Adjust this command depending on your OS and app location
            speak("Opening Google Chrome.")
        elif app_name.lower() == "notepad":
            subprocess.Popen(["notepad"])  # This is for Windows, adjust accordingly
            speak("Opening Notepad.")
        else:
            speak(f"Sorry, I can't open {app_name} at the moment.")
    except Exception as e:
        speak(f"An error occurred while trying to open {app_name}: {str(e)}")

if __name__ == '__main__':
    wishMe()
    userName()
    
    while True:
        order = takecommand().lower()
        
        if 'open project' in order:
            openProjectLink()
            
        elif 'open youtube' in order:
            openYouTube()
            
        elif 'open browser' in order:
            openBrowser()
            
        elif 'open' in order:  # To open any specified app like Chrome, Notepad, etc.
            app_name = order.replace('open', '').strip()
            openApp(app_name)
        
        elif 'exit' in order:
            speak("Goodbye!")
            break
            


