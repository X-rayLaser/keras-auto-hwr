# Introduction

keras-auto-hwr is a micro-framework for building a handwritten text 
recognition (HTR) system built on top of Keras and TensorFlow.

The central philosophy is to enable one to quickly train and test 
an RNN to perform HTR with minimum upfront efforts. One only needs 
to implement a thin API specifying how and where raw data for 
training should be fetched from. The framework's pipeline will 
automatically take care of steps such as pre-processing, 
normalization, converting target sequences into one-hot 
representation and more.

# Features
- minimal pre-processing
- automatic data normalization
- automatic encoding of text into a sequence of one-hot vectors
- built-in model (bidirectional LSTM model trained with CTC loss)
- train on GPU, predict on CPU
- saving/resuming training after any epoch
- interactive demo for testing a trained model live
- language-agnostic (work in progress)
- deploy a trained model for use with TensorFlow.js
- built-in Token Passing decoding algorithm

# Limitations
At the time of this writing, there are a few major limitations such as:
- No support for building an offline recognition system.
- Currently, the only model that is supported is the architecture 
based around connectionist temporal classification
- Fixed number of layers (2) and units (100)
- No built-in encoder-decoder network architecture

# Installation

Clone the repository
```
git clone <clone_url>
```

, create a virtualenv environment using Python 3
```
virtualenv --python='/path/to/python3/executable' venv
```
, activate the environment
```
. venv/bin/activate
```

, go inside the directory containing a hidden git folder and install 
all python modules required for the app
```
pip install -r requirements.txt
```

# Quick Start

## Training from scratch on Iam Online DB data set

First, download the data set, unzip it and place it under 
./datasets/iam_online_db/ folder. The layout of datasets folder should
be as follows:

iam_online_db/

--ascii-all/

--lineImages-all/

--lineStrokes-all/

--original-xml-all/

--original-xml-part/

Compile the data set:

```
python compile.py 'IamSource' 'default' --num_examples=7000
```

This command will compile a data set using IAM-OnDB data set, default
pre-processor. It will create training/validation/test data split
containing a total of 7000 examples.

Train a bidirectional LSTM model with CTC loss:
```
python train_ctc.py --cuda=true
```

Set flag "cuda" to use CuDNNLSTM implementation instead of LSTM.

You can suspend the script after any epoch and resume training later.

When training is complete, run an OCR demo script:
```
python demo/ocr_demo.py
```

Finally, deploy the model to be used in javascript:
```
python deploy.py
```

Your deployed model will be in ./weights/deployed/blstm folder.

## Training on arbitrary data set



Create a subclass of BaseSource class in the data/providers.py 
module by implementing constructor, get_sequences and 
__len__ methods. You are free to do anything in those methods. 
For instance, you can read the data in from the file or fetch 
them through the network.

The get_sequences method should return a 
generator that returns raw handwriting data and corresponding 
transcription text. Each handwriting should be a list of strokes. 
Each stroke is itself a list of the following format: 
(x1, y1, t1), (x2, y2, t2), ..., where x and y are pen position 
coordinates and t is the time respectively. Add necessary 
pre-processing if you need to.

Here is an example.

```
class MyDataProvider(BaseSource):
    def __init__(self, num_lines):
        pass

    def get_sequences(self):
        hwr = [
            [(23, 8, 323), (25, 9, 325)], # first stroke
            [(55, 2, 340), (58, 2, 380)]  # second stroke
        ]
        
        transcription = 'foobar'
        
        yield hwr, transcription

    def __len__(self):
        return 1
```

Next, compile a data set using a newly implemented data provider:
```
python compile.py 'MyDataProvider' 'default' --num_examples=1
```

## Testing pre-trained model

### Run demo script using Best Path Decoding algorithm
```
python demo/ocr_demo.py
```

### Run demo script using Token Passing Decoding algorithm

Change your current working directory to ./algorithms/cpp:
```
cd ./algorithms/cpp
```

Compile and link C++ code into binary:
```
./make_binary.sh
```

Return to the root path of the project
```
cd ../../
```

Build a dictionary and language model (e. g. containing 4000 words)
```
python build_lang_model.py dictionary --max_words=4000
```

Now you should be able to use Token Passing algorithm
for decoding RNN output in a demo script.
For that, simply set a "token_passing" flag to true:
```
python demo/ocr_demo.py --token_passing="true"
```

### About pre-trained model

The model has 1 bidirectional LSTM layer with 100 hidden units followed by a softmax layer 
containing 100 units (one unit per character plus special codes and blank code). 
The model was trained on the IAM On-Line Handwriting Database using 9450 examples of the 
form: handwriting patterns -> text lines. It was trained with Adam optimizer with a 
learning rate of 0.001 and CTC loss. The training was terminated after 31 epochs.

## Running tests

```
python -m unittest discover -s tests
```

# License

This software is licensed under MIT license (see LICENSE).

# References

[1] [Alex Graves et. al. Unconstrained Online Handwriting Recognition with Recurrent Neural Networks](https://papers.nips.cc/paper/3213-unconstrained-on-line-handwriting-recognition-with-recurrent-neural-networks.pdf)

[2] [S. Young, N. Russell, and J. Thornton.  Token passing: A simple conceptual model for connected speech recognition system](https://pdfs.semanticscholar.org/963c/f8f238745100ac6cc5cf730653a6e1849b62.pdf?_ga=2.58290915.813220193.1572590064-1733760606.1572590064)

[3] [Liwicki, M. and Bunke, H.: IAM-OnDB - an On-Line English Sentence Database Acquired from Handwritten Text on a Whiteboard](http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database/iam-on-line-handwriting-database#LiBu05-03)
