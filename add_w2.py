import time

import torch
import tqdm

model = torch.load("model.pt")
model["w2"] = torch.zeros((3, 4))

for i in tqdm.tqdm(range(10), desc="Pretend training run", unit="epoch"):
    time.sleep(0.1)

torch.save(model, "model.pt")
