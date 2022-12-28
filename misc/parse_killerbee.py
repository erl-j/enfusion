#%%
import glob
import os
#%%

DATA_DIR = "data/KillerBee samples"

# formats
formats = ["wav", "mp3", "ogg", "flac", "aiff", "aif", "aifc"]
path=os.path.join(DATA_DIR, f"**/*.*")
filepaths = glob.glob(path, recursive=True)

filepaths = [f for f in filepaths if f.split(".")[-1] in formats]

#%%

# def write_dir_structure(path, indent=0):
#     if os.path.isdir(path):
#         dir_name = os.path.basename(path)
#         dir_dict = {dir_name: {}}
#         for f in os.listdir(path):
#             dir_dict[dir_name].update(
#                 write_dir_structure(os.path.join(path, f), indent + 1)
#             )
#         return dir_dict
#     else:
#         return {}

# import json

# with open("data/killerbee_dir_structure.json", "w") as f:
#     json.dump(write_dir_structure(DATA_DIR), f, indent=4)
#%%

FOLDER_DISALLOWLIST = set(["Loops","__MACOSX"])

import json
# open kilerbee dir structure
with open("data/killerbee_dir_structure.json", "r") as f:
    dir_structure = json.load(f)

for f in filepaths:
    f= "/".join(f.split("/")[1:])
    # check if root folder is in disallowlist
    if os.path.basename(os.path.dirname(f)) in FOLDER_DISALLOWLIST:
        continue
    else:
        strings = f.split("/")
        caption_strings = []
        # include filename = strings[-1]
        filename=strings[-1]

        # traverse dir structure and add strings to caption as long as they are in the dir structure object

        local_dir_structure = dir_structure
        for i in range(len(strings)):
            if strings[i] in local_dir_structure:
                caption_strings.append(strings[i])
                local_dir_structure = local_dir_structure[strings[i]]
            else:
                break

        caption_strings.append(filename)

        caption = " ".join(caption_strings)

        print(caption)

        print(f)
        



        








# %%
