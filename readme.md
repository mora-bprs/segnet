# bin picking robot : enet implementation
this repo tracks the progress regarding the following,
- ENet research paper implementation locally
- ENet research paper implementation in google colab
- ENet optimizations for boxes
- Image instance segmentation integration with ENet
- Dataset tracking


## local setup in vscode
1. setup python venv and select it as the kernel
```sh
python3 -m venv .venv
source .venv/bin/activate
# or
source .venv/bin/activate.fish # depending on your shell
```
2. install required packages `pip install -r requirements.txt`
3. get the dataset in the directory
4. start rocking?

### References
All these cool people made these free stuff to make our project less taxing.
1. link to the paper: https://arxiv.org/pdf/1606.02147.pdf
2. reference repository: https://github.com/iArunava/ENet-Real-Time-Semantic-Segmentation