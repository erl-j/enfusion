import torch
import numpy as np
import os
import PIL
from PIL import Image
import json
from tqdm import tqdm

from encodec_processor import embedding2image,image2embedding

from sklearn.preprocessing import RobustScaler,MinMaxScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

class DrumfusionDataset(torch.utils.data.Dataset):
    def __init__(self):

        self.data = torch.load("artefacts/drums_data.pt")
        self.embeddings = torch.concat(
            [self.data[i]["encoded_frames_embeddings"] for i in range(len(self.data))], dim=0
        )

        self.scaler = Pipeline([('scaler', RobustScaler(quantile_range=(25, 75), with_centering=False, with_scaling=True)), ('minmax', MinMaxScaler(feature_range=(0, 1)))])
        self.scaler.fit(self.embeddings.permute([0,2,1]).reshape(-1,self.embeddings.shape[1]))

        # save pickled scaler to artefacts
        torch.save(self.scaler, "artefacts/scaled_aesd_dataset/scaler.pt")
    
        self.images = self.embedding2image(self.embeddings)


    def embedding2image(self,embeddings):
        # collapse batch and time dimension
        reshaped_embeddings = embeddings.permute([0,2,1]).reshape(-1,embeddings.shape[1])
        rescaled_embeddings = torch.tensor(self.scaler.transform(reshaped_embeddings))
        # reshape back to batch and time dimension
        images = rescaled_embeddings.reshape(-1,embeddings.shape[2],embeddings.shape[1]).permute([0,2,1])

        images = torch.nn.functional.interpolate(
            images[:, None, ...], size=(512, 512), mode="bilinear"
        )[:, 0, ...]

        images = images[...,None].repeat(1,1,1,3)*255.0
        return images

    def image2embedding(self,images):
        images = torch.mean(images, dim=-1)/255.0
        images = torch.nn.functional.interpolate(
            images[:, None, ...], size=(128, 150), mode="bilinear"
        )[:, 0, ...]
        
        # collapse batch and time dimension
        reshaped_images = images.permute([0,2,1]).reshape(-1,images.shape[1])
        rescaled_images = torch.tensor(self.scaler.inverse_transform(reshaped_images))
        # reshape back to batch and time dimension
        embeddings = rescaled_images.reshape(-1,images.shape[2],images.shape[1]).permute([0,2,1])
        return embeddings
        
    def __len__(self):
        return len(self.data)

    def export(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        metadata = []
        for i in tqdm(range(len(self.data))):
            image = self[i]["image"]
            text = self[i]["text"]
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
        return {"image":image, "text":text}

    

    
if __name__ == "__main__":

    dataset = DrumfusionDataset()

    dataset.export("artefacts/scaled_aesd_dataset")

    hist = np.histogram(dataset.images.numpy().reshape(-1), bins=256, range=(0, 255))
    plt.plot(hist[1][:-1], hist[0])
    print(dataset.images[0])
    plt.savefig("artefacts/histogram.png")