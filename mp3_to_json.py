# import whisper
# import json
# import os

# model = whisper.load_model("large-v2")

# audios = os.listdir("audios")

# for audio in audios: 
#     if("_" in audio):
#         number = audio.split("_")[0]
#         title = audio.split("_")[1][:-4]
#         print(number, title)
#         result = model.transcribe(audio = f"audios/{audio}", 
#         # result = model.transcribe(audio = f"audios/sample.mp3", 
#                               language="hi",
#                               task="translate",
#                               word_timestamps=False )
        
#         chunks = []
#         for segment in result["segments"]:
#             chunks.append({"number": number, "title":title, "start": segment["start"], "end": segment["end"], "text": segment["text"]})
        
#         chunks_with_metadata = {"chunks": chunks, "text": result["text"]}

#         with open(f"jsons/{audio}.json", "w") as f:
#             json.dump(chunks_with_metadata,f)
import whisper
import json
import os

# Load Whisper model
model = whisper.load_model("large-v2")

# Make sure folders exist
os.makedirs("jsons", exist_ok=True)

# List all audios
audios = os.listdir("audios")

for audio in audios:
    if "_" in audio:
        number = audio.split("_")[0]
        title = audio.split("_")[1][:-4]
        print(number, title)

        # Transcribe audio
        result = model.transcribe(
            audio=f"audios/{audio}",
            language="hi",
            task="translate",
            word_timestamps=False
        )

        # Create list of chunks
        chunks = []
        for segment in result["segments"]:
            chunks.append({
                "number": number,
                "title": title,
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip()
            })

        # Add metadata
        chunks_with_metadata = {
            "audio_file": audio,
            "chunks": chunks,
            "full_text": result["text"].strip()
        }

        #  Clean file name for safe saving
        safe_name = audio.replace("#", "").replace(" ", "_").replace("-", "_")

        #  Save JSON file
        with open(f"jsons/{safe_name}.json", "w", encoding="utf-8") as f:
            json.dump(chunks_with_metadata, f, ensure_ascii=False, indent=4)

print(" All audio files processed and JSONs saved successfully.")
