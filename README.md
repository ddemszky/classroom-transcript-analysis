# NCTE Classroom Transcript Analysis

Please cite the following when using the transcripts:
> Demszky, D., & Hill, H. (2022). [The NCTE transcripts: A dataset of elementary math classroom transcripts](https://arxiv.org/pdf/2211.11772.pdf). _arXiv preprint arXiv:2211.11772_.


**EACH user** who would like to access the dataset should fill out this form: https://forms.gle/1yWybvsjciqL8Y9p8. Once you fill it out, the Google Drive folder will be shared with you automatically.


The dataset contains the following files:

1. `single_utterances.csv`: A csv file containing all utterances from the transcript dataset. The `OBSID` column represents the unique ID for the transcript, and the `NCTETID` represents the teacher ID, which are mappable to metadata. `comb_idx` represents a unique ID for each utterance (concatenation of `OBSID` and `turn_idx`), which is mappable to turn-level annotations.
2. `student_reasoning.csv`: Turn-level annotations for `student_reasoning`. The annotations are binary. 
3. `paired_annotations.csv`: Turn-level annotations for `student_on_task`,	`teacher_on_task`,	`high_uptake`,	`focusing_question`, using majority rater labels. The annotation protocol is included under the `coding schemes` folder.

The transcripts are associated with metadata, including observation scores, value added measures and student questionnaire responses. The metadata and additional documentation are available on [ICPSR](https://www.icpsr.umich.edu/web/ICPSR/studies/36095). You can use the OBSID variable and the NCTETID variables to map transcript data to the metadata.

**Issues with transcripts:** Certain transcripts have issues with respect to speaker assignment. Namely, student utterances may be labeled as teacher utterances and vice versa. The `transcript_issues.txt` includes a list of OBSIDs that we recommend excluding from your analyses. If you encounter issues with other transcripts, please feel free to make a pull request or email [Dora](mailto:ddemszky@stanford.edu).


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

For any questions about the dataset, please email Dora at ddemszky@stanford.edu.

