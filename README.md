# Transformer Chatbot 
A simple quesstion answering bot based on transformer architecture. 
![arch](https://www.tensorflow.org/images/tutorials/transformer/transformer.png)

## Getting Started
To get started training simply run `python train.py`. You can adjust the hypter parameters
in the file.

### Dependencies
You have to download the following packages
- `tensorflow-gpu 2.2.0`
- `pandas 1.04`
- `sklearn 0.23.1`
- `numpy 1.18.5`

### Data Prep

To prepare for trainning, you need to download raw data from stack exchange. Fortunately,
it is an easy step by executing `download_question_links`, which will crawl the links
of questions. Then you execute `download_question_data`,which will download and parse
those links. This can take an average of 10 hours. 

This script will generate a `data.csv` file that contains about 50k question and
answers from various fields.

## Test the Model

Once the training is done, you can run evluation step in the `evaluate.py`. If you 
want to test a question, just execute `python evaluate.py "Why do some people get paid more?"`


## Acknowledgments

* The `model.py` is almost directly copied from the   [Tensorflow tutorial page.](https://www.tensorflow.org/tutorials/text/transformer) 
