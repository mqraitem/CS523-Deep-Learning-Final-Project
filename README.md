# facebok_hateful_memes
Challenge link: [link](https://www.drivendata.org/competitions/64/hateful-memes/page/205/)

### Data:
We get the image feature vectors using a Faster RCNN model from the following repo: [link]( https://github.com/airsplay/py-bottom-up-attention).
Once this is done, we convert the image features to a hdf5 file using convert_images.py under utils. The hdf5 data can be found on SCC under the folder mqraitem.

### Models:
We provide two versions of the model:
  * Text GLoVE: fbhm_text_only
  * Visual GLoVE: fbhm_text_vision.

### Running the code:
To train a model, please decide on a model name and then create a folder with the name in both logs and runs. Then run:

$ python train.py --model_name [your model name] --build_vocab [1]

You don't need to build vocab more than once so you can set it to zero in the second run.

### Evaluation:
To evaluate the code, please run:

$ python test.py --model_path ../runs/[your model name]/[desired epoch]

### Computational requirements:
The model is light weight. It can be trained and evalulated on a cpu.  
