from fastapi import FastAPI, UploadFile, File, HTTPException, Query
import uuid
import aiofiles
import os

from pydub import AudioSegment

# Imports
from tflite_support.task import audio
from tflite_support.task import core
from tflite_support.task import processor
import os

app = FastAPI()

@app.get("/")
def health_check():
    return "OK"


def mp4_2_wav(filename):
    """Convert mp4 to wav.
    Args:
        filename: mp4 file path.
    Returns:
        wav file path.
    """
    # convert mp4 to wav
    wav_file_path = filename.replace('.mp4', '.wav')
    
    sound = AudioSegment.from_file(filename, format="mp4")
    new_sample_rate = 44100
    sound = sound.set_frame_rate(new_sample_rate)
    sound.export(wav_file_path, format="wav")
    num_channels = sound.channels
    return wav_file_path, num_channels

def stereo2mono(filename):
    # Open the stereo sound
    stereo_sound = AudioSegment.from_wav(filename)
    
    # Calling the split_to_mono() method on the stereo sound will return a tuple
    mono_audios = stereo_sound.split_to_mono()
    # Export the two mono channels as separate wav files
    mono_left = mono_audios[0].export(filename.replace('.wav', '_left.wav'), format="wav")
    return filename.replace('.wav', '_left.wav')

@app.post("/predict")
async def get_file(file: UploadFile = File(...), keyword=Query('')):
    if len(file.filename) <= 0:
        raise HTTPException(status_code=400, detail="File not found")
    
    filename = str(uuid.uuid4()) + file.filename
    filename = 'src/data/' + str(filename)
    async with aiofiles.open(filename, 'wb') as out_file:
        while content := await file.read(1024):  # async read chunk
            await out_file.write(content)  # async write chunk

    # Initialization
    model_path = "./converted_tflite/soundclassifier_with_metadata.tflite"
    base_options = core.BaseOptions(file_name=model_path)
    classification_options = processor.ClassificationOptions(max_results=2)
    options = audio.AudioClassifierOptions(base_options=base_options, classification_options=classification_options)
    classifier = audio.AudioClassifier.create_from_options(options)

    # Run inference
    mp4_audio_path = filename
    wav_audio_path, num_channels = mp4_2_wav(mp4_audio_path)
    if num_channels == 1:
        pass
    elif num_channels == 2:
        wav_audio_path = stereo2mono(wav_audio_path)
    else:
        raise HTTPException(status_code=400, detail="Audio channel not supported")
    
    audio_file = audio.TensorAudio.create_from_wav_file(wav_audio_path, classifier.required_input_buffer_size)
    audio_result = classifier.classify(audio_file)
    os.remove(mp4_audio_path)
    os.remove(wav_audio_path)

    return audio_result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
