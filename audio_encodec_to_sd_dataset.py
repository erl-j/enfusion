import torch
import numpy as np
import os
import PIL
from PIL import Image
import json
from tqdm import tqdm

class DrumfusionDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.MIN_VALUE=-20
        self.MAX_VALUE=18
        self.data = torch.load("artefacts/drums_data.pt")
        self.embeddings = torch.concat(
            [self.data[i]["encoded_frames_embeddings"] for i in range(len(self.data))], dim=0
        )
        self.images = self.embedding2image(self.embeddings)

       

    def __len__(self):
        return len(self.data)

    def embedding2image(self, embeddings):
        # standard scale
        embeddings = (embeddings - self.MIN_VALUE) / (self.MAX_VALUE - self.MIN_VALUE)
        # upscale to batch by 512 by 512
        embeddings = torch.nn.functional.interpolate(
            embeddings[:, None, ...], size=(512, 512), mode="bilinear"
        )[:, 0, ...]
        embeddings = (embeddings*255)[..., None].repeat([1, 1, 1, 3])
        return embeddings

    def image2embedding(self, images):
        # turn into 0-1
        images = images / 255
        embeddings = torch.mean(images, dim=-1)
        embeddings = torch.nn.functional.interpolate(
            embeddings[:, None, ...], size=(128, 150), mode="bilinear"
        )[:, 0, ...]
        # turn into original scale
        embedding_hat = embedding_hat * (self.MAX_VALUE - self.MIN_VALUE) + self.MIN_VALUE
        return embedding_hat

    def export(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        metadata = []
        for i in tqdm(range(len(self.data))):
            image, text = self[i]
            image_path = os.path.join(output_dir, f"{i}.png")
            metadata.append({"file_name": f"{i}.png", "text": text})
            
            pil_image = Image.fromarray(image.numpy().astype(np.uint8))
            pil_image.save(image_path)
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)
        


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