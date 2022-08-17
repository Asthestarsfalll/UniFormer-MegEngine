# UniFormer-MegEngine
The MegEngine Implementation of UniFormer

## Usage

### Install Dependencies

```bash
pip install -r requirements.txt
```

If you don't want to compare the ouput error between the MegEngine implementation and PyTorch one, just ignore requirements.txt and install MegEngine from the command line:

```bash
python3 -m pip install --upgrade pip 
python3 -m pip install megengine -f https://megengine.org.cn/whl/mge.html
```

### Convert Weights

Convert trained weights from torch to megengine, the converted weights will be save in ./pretained/ , you need to specify the converte model architecture and path to checkpoint offered by official repo.

```bash
python convert_weights.py --model uniformer_base --ckpt /path/to/weights
```

### Compare

Use `python compare.py` .

By default, the compare script will convert the torch state_dict to the format that megengine need.

If you want to compare the error by checkpoints, you neet load them manually.

### Load From Hub

Import from megengine.hub:

Way 1:

```python
from functools import partial
import megengine.module as M
from megengine import hub

modelhub = hub.import_module(
    repo_info='asthestarsfalll/UniFormer-MegEngine:main', git_host='github.com')

# load UniFormer model and custom on you own
model = modelhub.UniFormer(
    depth=[3, 4, 8, 3],
    embed_dim=[64, 128, 320, 512], head_dim=64, mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(M.LayerNorm, eps=1e-6))

# load pretrained model
pretrained_model = modelhub.uniformer_small(pretrained=True)
```

Way 2:

```python
from  megengine import hub

# load pretrained model 
model_name = 'uniformer_small'
pretrained_model = hub.load(
    repo_info='asthestarsfalll/UniFormer-MegEngine:main', entry=model_name, git_host='github.com', pretrained=True)
```

You can still load the model without pretrained weights like this:

```python
model = modelhub.mae_vit_large_patch16()
# or
model_name = 'uniformer_base'
model = hub.load(
    repo_info='asthestarsfalll/UniFormer-MegEngine:main', entry=model_name, git_host='github.com')
```

## TODO

- [ ] Down stream tasks maybe

## Reference

[The official implementation of UniFormer](https://github.com/Sense-X/UniFormer)
