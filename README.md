# CS523 Final Project [Fall 2020]
In this project, we repurpose the Vision-Language model [Pythia 0.1](https://arxiv.org/pdf/1807.09956.pdf) for the Hateful Meme Classification [Challenge](https://www.drivendata.org/competitions/64/hateful-memes/page/205/). We obtain strong results with a simple model that is easy to train. We provide two versions of the model: 

* text_only
* text_vision.

text_only is only trained on the textual representation of the meme. text_vision incorporates the visual element of the meme as well as the textual one. 

### Results: 


|   Test Set    |     AUROC     |
| ------------- | ------------- |
| text_only     | 65.1          |
| text_vision   | 71.3          | 


The combination of text and vision is the most effective!

### Data:
We get the image feature vectors using a Faster RCNN model from the following repo: [link]( https://github.com/airsplay/py-bottom-up-attention).
Once this is done, we convert the image features to a hdf5 file using convert_images.py under utils. 

### Running the code:
To train a model, please decide on a model name and then create a folder with the name in both logs and runs. Then run:

`python [fbhm_test_only/fbhm_test_vision]/train.py --model_name [your model name] --build_vocab [1]`

You don't need to build vocab more than once so you can set it to zero in the second run.

### Evaluation:
To evaluate the code, please run:

`python [fbhm_test_only/fbhm_test_vision]/test.py --model_path ../runs/[your model name]/[desired epoch]`
