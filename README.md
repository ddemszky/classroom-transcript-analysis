# Classroom Transcript Analysis

This repository contains code for the paper:
> Demszky, D. & Hill, H. C.. (2022). The NCTE Transcripts: A Dataset of Elementary Math Classroom Transcripts. _To appear._

**We are still working on the terms of use for the dataset, but we expect that soon it will be available to researchers. If you are interested in using this dataset for your research, please fill out this form: https://forms.gle/7d5vNPfx2gKZyu6m6**

We currently are only releasing data from **a randomly selected 80% of teachers**. The remaining 20% is reserved as a heldout set. We may share the heldout data at some point in the upcoming years.

The dataset contains the following files:

1. `single_utterances_release.csv`: A csv file containing all utterances from the transcript dataset. The `OBSID` column represents the unique ID for the transcript, and the `NCTETID` represents the teacher ID, which are mappable to metadata. `comb_idx` represents a unique ID for each utterance (concatenation of `OBSID` and `turn_idx`), which is mappable to turn-level annotations.
2. `transcript_metadata_release.csv`: A csv file containing transcript metadata. The columns are described in [this spreadsheet](https://docs.google.com/spreadsheets/d/19PmekP0hAyzGdHyzrLy-Dr1b-CUgFC3vOpUwvcQcxog/edit#gid=0). More detailed documentation of the metadata as well as additional metadata are available on [ICPSR](https://www.icpsr.umich.edu/web/ICPSR/studies/36095).
3. `student_reasoning_release.csv`: Turn-level annotations for `student_reasoning`. The annotations are binary. 
4. `paired_annotations_release.csv`: Turn-level annotations for `student_on_task`,	`teacher_on_task`,	`high_uptake`,	`focusing_question`, using majority rater labels. The annotation protocol is included under the `coding schemes` folder.


## Train a Turn-Level Classifier
You can use the `run_classifier.py` script to train turn-level classifiers like the ones we describe in the paper.

Set up the virtual environment:
1. Create virtual environment: `python3 -m venv venv`
2. Activate virtual environment: `source venv/bin/activate`
3. Install requirements `$ pip3 install -r requirements.txt`. You might need to use different pytorch versions depending on whether you are using a GPU or a CPU. We recommend using a GPU for training.

### Run fine-tuning

The following script runs training for the `student_on_task` discourse move, using 90% of all annotations (`dev_split_size`=0.1), while balancing out 0 and 1 labels while training. It also runs predictions on all the data once the model finished training. You can tailor the parameters to your own setting easily (e.g. choose a different discourse move). 
```
python run_classifier.py \
--train \
--train_data=data/paired_annotations_release.csv \
--dev_split_size=0.1 \
--num_train_epochs=5 \
--text_cols=student_text \
--label_col=student_on_task \
--predict \
--predict_data=data/paired_utterances.csv \
--predict_index_col=exchange_idx \
--balance_labels
```
### Run cross validation

The following script runs 5-fold cross-validation for the `focusing_question` discourse move, while balancing out 0 and 1 labels while training. It also runs predictions on all the data once the model finished training. 
```
python run_classifier.py \
--cv \
--train_data=data/paired_annotations.csv \
--num_train_epochs=5 \
--text_cols=student_text,teacher_text \
--label_col=focusing_question \
--predict_index_col=exchange_idx \
--balance_labels
```


