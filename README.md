# Dialog_Extension
A unified Dialog System including Chitchat，TOD，QA and CRS

## Environment setting
Our python version is 3.7.11

The package can be installed by running the following command.

```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Data Preprocessing
### ChitChat
FusedChat  https://github.com/tomyoung903/FusedChat/tree/main/data
```
cd data/CC/FusedChat
python preFusedChat.py
```
Ubuntu   https://github.com/npow/ubottu
```
cd data/CC/Ubuntu
python preUbuntu.py 
```
### CQA
Squad2.0   https://rajpurkar.github.io/SQuAD-explorer/
```
cd data/CQA/Squad
python preSquad.py 
```
coQA  https://stanfordnlp.github.io/coqa/
```
cd data/CQA
python precoQA.py
```
### TOD
multiwoz2.0   https://github.com/budzianowski/multiwoz
```
python preprocess.py
```
### CRS
ReDial   https://redialdata.github.io/website/
```
cd data/CRS/ReDial
python preRedial.py
```
For training with the cat data, you should cat the datasets firstly
```
cd data/util
python datasetcat.py
```
## Train
Our code support DDP
```
bash train.sh
```
the parameter data_type indicate the training data type

## Test
```
bash test.sh
```

## Acknowledgements

This code is based on the released code (https://github.com/bepoetree/MTTOD) for "Improving End-to-End Task-Oriented Dialogue System with A Simple Auxiliary Task",  

