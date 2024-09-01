import os
import glob 
import random 

import jsonlines

librispeech_dir = "./dataset/LibriSpeech/*"
enroll_path = "./dataset/enroll_audios/all"

if not os.path.exists(enroll_path):
    os.makedirs(enroll_path)

speaker_dirs = glob.glob(f"{librispeech_dir}/*")
new_files = []
for speaker_dir in speaker_dirs:
    if not "train-" in speaker_dir and not "test-" in speaker_dir and not "dev-" in speaker_dir:
        continue
    speaker_id = os.path.basename(speaker_dir)
    enroll_speaker_dir = os.path.join(enroll_path, speaker_id)
    if not os.path.exists(enroll_speaker_dir):
        os.makedirs(enroll_speaker_dir, exist_ok=True)
    # Check whether each speaker in librispeech_dir exists in enroll_path
    if len(glob.glob(f"{enroll_speaker_dir}/*.wav")) == 0:
        # Copy an audio file from librispeech_dir to enroll_speaker_dir
        flac_files = glob.glob(f"{speaker_dir}/*/*.flac")
        flac_file = random.choice(flac_files)
        new_flac_file = os.path.join(enroll_speaker_dir, os.path.basename(flac_file))
        print(new_flac_file)
        os.system(f"cp {flac_file} {new_flac_file}")
        # Record the name of the flac_file
        new_files.append(flac_file)
    else:
        print(glob.glob(f"{enroll_speaker_dir}/*.wav"))
    
    # Check whether the voice naming of the speaker in enroll_path is {speaker_id}.wav
    enroll_wav_files = glob.glob(f"{enroll_speaker_dir}/*")
    for enroll_wav_file in enroll_wav_files:
        if os.path.basename(enroll_wav_file).split(".")[0] != speaker_id:
            print(f"{enroll_wav_file} not match")
            # Convert to wav format, keep only the speaker_id in the name
            wav_file = os.path.join(enroll_speaker_dir, speaker_id + ".wav")
            os.system(f"ffmpeg -i {enroll_wav_file} {wav_file}")
            os.system(f"rm -rf {enroll_wav_file}")

# Record the original name of the enroll wav
with open(f"{enroll_path}/enrolled_wavs.txt", "w") as f:
    f.write("\n".join(new_files))