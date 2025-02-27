import sounddevice as sd
import numpy as np
import whisper  
import ollama
import pyttsx3  
import sys  


whisper_model = whisper.load_model("base")  


model_name = "mistral"  


engine = pyttsx3.init()
voices = engine.getProperty("voices")
engine.setProperty("voice", voices[1].id)  


nombre_asistente = "ELISA"

def grabar_audio(duracion=5, tasa_muestreo=16000):
    """Graba audio desde el micrófono."""
    print("Grabando...")
    grabacion = sd.rec(int(duracion * tasa_muestreo), samplerate=tasa_muestreo, channels=1, dtype='float32')
    sd.wait()  
    print("Grabación terminada.")
    return grabacion.flatten()

def transcribir_audio(audio, tasa_muestreo=16000):
    """Transcribe el audio a texto utilizando Whisper de OpenAI."""
 
    audio = audio.astype(np.float32)
    
   
    resultado = whisper_model.transcribe(audio)
    
    return resultado["text"]

def generar_respuesta(texto):
    """Genera una respuesta utilizando Mistral a través de Ollama."""
 
    prompt = (
        f"Eres {nombre_asistente}, un asistente virtual amigable, servicial y con un toque de humor. "
        f"Responde al siguiente mensaje de manera clara y concisa: {texto}"
    )
    respuesta = ollama.generate(model=model_name, prompt=prompt)
    return respuesta["response"]

def hablar(texto):
    """Convierte el texto en voz femenina."""
    print(f"{nombre_asistente}: {texto}")  
    engine.say(texto)
    engine.runAndWait()

def despedir():
    """Función para despedirse y finalizar el programa."""
    hablar("¡Hasta luego! Gracias por hablar conmigo. 😊")
    sys.exit()  

def asistente_virtual():
    print(f"{nombre_asistente}: Hola, soy {nombre_asistente}, tu asistente virtual. ¿En qué puedo ayudarte hoy?")
    while True:
        try:
            
            audio = grabar_audio()
            
            
            texto = transcribir_audio(audio)
            print(f"Usuario: {texto}")
            
            
            if any(frase in texto.lower() for frase in ["chao elisa gracias", "hasta luego elisa", "gracias elisa, hasta luego"]):
                despedir()
            
            
            respuesta = generar_respuesta(texto)
            
            
            hablar(respuesta)
            
        except KeyboardInterrupt:
            print(f"{nombre_asistente}: Hasta luego. ¡Fue un placer ayudarte!")
            break

if __name__ == "__main__":
    asistente_virtual()
