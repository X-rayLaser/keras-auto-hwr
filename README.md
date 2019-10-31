# Introduction

A micro-framework intended to easily train a system that performs
recognition of unconstrained hand-written text. 

# Limitations

Currently, the repo can only be used to train an on-line recognition
system.

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

## Iam Online DB data set

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

# License

This software is licensed under MIT license (see LICENSE).
