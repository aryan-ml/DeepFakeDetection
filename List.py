import os

# 1. POINT THIS TO YOUR DEEPFAKES FOLDER
# (Make sure this path matches where your 150 videos actually are)
deepfake_path = "./data/raw/manipulated_sequences/Deepfakes/c23/videos"

needed_ids = set()

if os.path.exists(deepfake_path):
    for filename in os.listdir(deepfake_path):
        if filename.endswith(".mp4"):
            # filename is "045_889.mp4"
            parts = filename.split('_') # splits into ['045', '889.mp4']
            
            # Clean up the names to get just the ID string
            id1 = parts[0]
            id2 = parts[1].replace('.mp4', '')
            
            needed_ids.add(id1)
            needed_ids.add(id2)
            
    # Print the list formatted for the next step
    print(f"found {len(needed_ids)} unique original IDs needed.")
    print("COPY THE LINE BELOW:")
    print(list(needed_ids))
else:
    print("Error: Could not find folder. Check the path in the script.")