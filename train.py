import time

import torch
import tqdm

model = {"w1": torch.ones((2, 3)), "b1": torch.zeros((3,))}

for i in tqdm.tqdm(range(10), desc="Pretend training run", unit="epoch"):
    time.sleep(0.1)

torch.save(model, "model.pt")
