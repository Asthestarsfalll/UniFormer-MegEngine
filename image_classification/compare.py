import time

import megengine as mge
import numpy as np
import torch

from models.torch_model import uniformer_base as torch_uniformer_base
from models.torch_model import uniformer_base_ls as torch_uniformer_base_ls
from models.torch_model import uniformer_small as torch_uniformer_small
from models.torch_model import \
    uniformer_small_plus as torch_uniformer_small_plus
from models.torch_model import \
    uniformer_small_plus_dim64 as torch_uniformer_small_plus_dim64
from models.uniformer import (uniformer_base, uniformer_base_ls, uniformer_small,
                            uniformer_small_plus, uniformer_small_plus_dim64)
from convert_weights import convert

# mge_model = uniformer_small(True)
mge_model = uniformer_small_plus(True)
# mge_model = uniformer_small_plus_dim64(True)

# torch_model = torch_uniformer_small()
torch_model = torch_uniformer_small_plus()
# torch_model = torch_uniformer_small_plus_dim64()

print()
print("Can not download the torch weights automatically")
print("So we need to convert state dict instead of loading trained weights.")
print("Convert the state dict of torch to megengine...")

state_dict = torch_model.state_dict()

new_dict = convert(torch_model, state_dict)
print("Convert down, load")
print()

mge_model.load_state_dict(new_dict)
# torch_model.load_state_dict(torch.load('./image_classification/uniformer_small_in1k.pth', map_location='cpu')['model'])
# torch_model.load_state_dict(torch.load('./image_classification/uniformer_small_plus_in1k.pth', map_location='cpu')) # no 'model'
# torch_model.load_state_dict(torch.load('./image_classification/uniformer_small_plus_dim64_in1k.pth', map_location='cpu')['model'])

# mge_model.load_state_dict(mge.load("./pretrained/uniformer_small_plus_dim64.pkl"))

mge_model.eval()
torch_model.eval()

torch_time = meg_time = 0.0

def test_func(mge_out, torch_out):
    result = np.isclose(mge_out, torch_out, rtol=1e-3)
    ratio = np.mean(result)
    allclose = np.all(result) > 0
    abs_err = np.mean(np.abs(mge_out - torch_out))
    std_err = np.std(np.abs(mge_out - torch_out))
    return ratio, allclose, abs_err, std_err


def softmax(logits):
    logits = logits - logits.max(-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(-1, keepdims=True)

print("Begin testing")
for i in range(15):
    results = []
    inp = np.random.randn(2, 3, 224, 224)
    mge_inp = mge.tensor(inp, dtype=np.float32)
    torch_inp = torch.tensor(inp, dtype=torch.float32)

    if torch.cuda.is_available():
        torch_inp = torch_inp.cuda()
        torch_model.cuda()

    st = time.time()
    mge_out = mge_model(mge_inp)
    meg_time += time.time() - st

    st = time.time()
    torch_out = torch_model(torch_inp)
    torch_time += time.time() - st

    if torch.cuda.is_available():
        torch_out = torch_out.detach().cpu().numpy()
    else:
        torch_out = torch_out.detach().numpy()
    mge_out = mge_out.numpy()
    mge_out = softmax(mge_out)
    torch_out = softmax(torch_out)
    ratio, allclose, abs_err, std_err = test_func(mge_out, torch_out)
    results.append(allclose)
    print(f"Result: {allclose}, {ratio*100 : .4f}% elements is close enough\n which absolute error is  {abs_err} and absolute std is {std_err}")

assert all(results), "not aligned"

print(f"meg time: {meg_time}, torch time: {torch_time}")