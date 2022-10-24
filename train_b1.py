import time

import torch
import tqdm

model = torch.load("model.pt")
model["b1"] = torch.ones((3,))

for i in tqdm.tqdm(range(10), desc="Pretend training run", unit="epoch"):
    time.sleep(0.1)

torch.save(model, "model.pt")
