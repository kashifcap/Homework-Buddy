# Homework-Buddy(A part of I.E.E.E Megaproject)

## About our work
Implementation of handwriting generation with use of recurrent neural networks in tensorflow. Based on Alex Graves paper (https://arxiv.org/abs/1308.0850).

## Using our HandwrittenModel package

#### 1. Download dataset
First you need to download dataset. This requires you to register on [this page](http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database) ("Download" section). After registration you will be able to download the [data/original-xml-part.tar.gz](http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database/download-the-iam-on-line-handwriting-database).

#### 2. Preprocessing data
```
python Datapreprocess.py
```
This scipt searches local directory for `xml` files with handwriting data and does some preprocessing like normalizing data and spliting strokes in lines. As a result it should create `data` directory with preprocessed dataset.

#### 3. Train model
```
python train.py
```

This will launch training with default settings (for experimentation look at `argparse` options). By default it creates `summary` directory with separate `experiment` directories for each run.

#### 4. Handwritting generation

You can use the main function of HandwrittenModel package in your application to get started.It uses a `text.txt` file to generate the handwritten file and saves it in pdf format named `Handwritten.pdf`.

```
python
>>>from HandwrittenModel import main
>>>main()
```

## Using our web interface and getting your assignments done with ease!

#### Activating the virtualenvironment

1. In linux/Mac
```
source venv/bin/activate
```

2. In windows
```
source venv/scripts/activate
```

#### Downloading the required packages

```
pip install -r requirements.txt
```

#### Running the flask app

```
python app.py
```

Now open your browser and visit (http://127.0.0.1:5000)
This will bring you the the Handwritting-Buddy homepage.
Follow the instructions and complete your assignments in no time!