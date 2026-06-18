import os
import subprocess
import pandas as pd
import time

clips_folder = "data/audio_dataset"
labels_csv = "data/labels.csv"
repeats = 10

# Load existing labels
if os.path.exists(labels_csv):
    labels_df = pd.read_csv(labels_csv)
    labeled = set(labels_df["filename"])
else:
    labels_df = pd.DataFrame(columns=["filename", "label"])
    labeled = set()

# Get all clips
all_clips = sorted([f for f in os.listdir(clips_folder) if f.lower().endswith(".wav")])
unlabeled = [f for f in all_clips if f not in labeled]

print(f"Total clips: {len(all_clips)}")
print(f"Already labeled: {len(labeled)}")
print(f"Remaining to label: {len(unlabeled)}")

# YOU label: clips 049-068
# YOUR TEAMMATE labels: clips 069-088
start_clip = 70  # Change this: You=49, Teammate=69
end_clip = 88    # Change this: You=68, Teammate=88

clips_to_label = [f"clip_{i:03d}.wav" for i in range(start_clip, end_clip+1)]
clips_to_label = [c for c in clips_to_label if c in unlabeled]

print(f"\nYou will label {len(clips_to_label)} clips")
print(f"Starting from clip_{start_clip:03d}.wav to clip_{end_clip:03d}.wav")
input("Press Enter to start...")

for clip in clips_to_label:
    path = os.path.join(clips_folder, clip)
    print(f"\n{'='*50}")
    print(f"Playing {clip} ({repeats} times)...")
    print(f"{'='*50}")
    
    for i in range(repeats):
        print(f"Repetition {i+1}/{repeats}...", end='\r')
        subprocess.call(['afplay', path])
        time.sleep(0.01)
    
    print("\n")
    label = input("Does it transform into song? 1=yes, 0=no: ").strip()
    while label not in {"0", "1"}:
        label = input("Enter 1 (transforming) or 0 (non-transforming): ").strip()
    
    labels_df = pd.concat(
        [labels_df, pd.DataFrame([[clip, int(label)]], columns=["filename", "label"])],
        ignore_index=True
    )
    labels_df.to_csv(labels_csv, index=False)
    print(f"✓ Labeled {clip} as {label}")
    print(f"Progress: {len(labels_df)}/{len(all_clips)} total clips labeled")

print("\n" + "="*50)
print("DONE! Your portion is complete.")
print(f"Total labeled so far: {len(labels_df)}/{len(all_clips)}")
print("="*50)