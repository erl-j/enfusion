import torch
import numpy as np
import os
import PIL
from PIL import Image
import json
from tqdm import tqdm

from encodec_processor import embedding2image,image2embedding

class DrumfusionDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.MIN_VALUE=-20
        self.MAX_VALUE=18
        self.data = torch.load("artefacts/drums_data.pt")
        self.embeddings = torch.concat(
            [self.data[i]["encoded_frames_embeddings"] for i in range(len(self.data))], dim=0
        )
        self.images = embedding2image(self.embeddings,min_value=self.MIN_VALUE,max_value=self.MAX_VALUE)

    def __len__(self):
        return len(self.data)

    def export(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        metadata = []
        for i in tqdm(range(len(self.data))):
            image, text = self[i]
            image_path = os.path.join(output_dir, f"{i}.png")
            metadata.append({"file_name": f"{i}.png", "text": text})
            
            pil_image = Image.fromarray(image.numpy().astype(np.uint8))
            pil_image.save(image_path)
        with open(os.path.join(output_dir, "metadata.jsonl"), "w") as f:
            # write json lines file
            for line in metadata:
                f.write(json.dumps(line) + "\n")
        


    def __getitem__(self, idx):
        # fp = self.data[idx]["filepath"]
        # embedding = self.data[idx]["encoded_frames_embeddings"][0]
        image = self.images[idx]
        text_string = self.data[idx]["folder"] + self.data[idx]["filename"].split(".")[0]

        categories = ["kick", "snare","rim", "hihat", "tom", "cymbal","ride","clap","crash","percussion","other","fx"]

        aliases = {
            "kick": ["kick", "kck","bass","bd"],
            "snare":["snr","snare","sn","sd"],
            "rim":["rim","rimshot"],
            "hihat":["hihat","hh","hats","hat","hi-hat","hihat","hi-hats","open","closed","chh","ohh"],
            "crash":["crash","cr"],
            "ride":["ride","rd"],
            "cymbal":["cymbal","cym","splash"],
            "tom":["tom","tm"],
            "clap":["clap","cl","clp"],
            "percussion":["percussion","perc","agogo","bongo","cabasa","castanets","clave","conga","cowbell","guiro","maracas","marimba","shaker","tambourine","triangle","vibraslap","woodblock"],
            "fx":["effect","fx"],
            "other":[]
        }

        text=""
        for category in categories:
            for alias in aliases[category]:
                if alias in text_string:
                    text = category
                    break
            else:
                continue
            break

        if text == "":
            text = "other"

        # convert to image
        return (image, text)

    

    
if __name__ == "__main__":

    dataset = DrumfusionDataset()

    dataset.export("artefacts/aesd_dataset")