# converts the video to mp3

import os
import subprocess

files = os.listdir("LLM videos")
for file in files:
    
    tutorial_number = file.split("#")[1].split("-")[0].strip()
    file_name = file.split(" - ")[0]
    print(tutorial_number, file_name)
    subprocess.run(["ffmpeg", "-i",f"LLM videos/{file}", f"audios/{tutorial_number}_{file_name}.mp3"])
