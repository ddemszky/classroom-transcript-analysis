"""
python run_classifier.py

To run cross validation:
python run_classifier.py \
--cv \
--train_data=[TRAIN CSV]] \
-num_train_epochs=5 \
--text_cols=text \
--label_col=[LABEL COLUMN] \
--predict_index_col=[INDEX COLUMN FOR STORING PREDICTIONS] \
--balance_labels
"""
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from argparse import ArgumentParser
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, train_test_split
from scipy.stats import pearsonr, spearmanr
import warnings
import pandas as pd
from sys import exit
import logging
import torch
warnings.filterwarnings("ignore")


def pearson_corr(preds, labels):
    return pearsonr(preds, labels)[0]

def spearman_corr(preds, labels):
    return spearmanr(preds, labels)[0]

def accuracy(preds, labels):
    return sum([p == l for p, l in zip(preds, labels)]) /len(labels)

def precision(preds, labels):
    return precision_score(y_true=labels, y_pred=preds)

def recall(preds, labels):
    return recall_score(y_true=labels, y_pred=preds)

def f1(preds, labels):
    return f1_score(y_true=labels, y_pred=preds)


def train(colname, train_df, eval_df, text_cols,
          output_dir, model="roberta", num_labels=2,
          num_train_epochs=10,
          train_batch_size=8, gradient_accumulation_steps=2,
          max_seq_length=512,
          cross_validate=False,
          balance_labels=True):
    print("Train size: %d" % len(train_df))
    print("Eval size: %d" % len(eval_df))

    print(train_df.head())
    print(eval_df.head())

    print("Is CUDA available? " + str(torch.cuda.is_available()))

    if balance_labels:
        most_common = train_df["labels"].value_counts().idxmax()
        print("Most common label is: %s" % most_common)
        most_common_df = train_df[train_df["labels"]==most_common]
        concat_list = [most_common_df]
        for label, group in train_df[train_df["labels"]!=most_common].groupby("labels"):
            concat_list.append(group.sample(replace=True, n=len(most_common_df)))
        train_df = pd.concat(concat_list)
        print("Train size: %d" % len(train_df))
        print(train_df["labels"].value_counts())

    # Shuffle training data
    train_df = train_df.sample(frac=1)
    save_dir = output_dir + "/" + colname + "_train_size=" + str(len(train_df))

    model_args = ClassificationArgs()
    model_args.reprocess_input_data = True
    model_args.overwrite_output_dir = True
    model_args.evaluate_during_training = True  # change if needed
    model_args.max_seq_length = int(max_seq_length / len(text_cols))
    model_args.num_train_epochs = num_train_epochs
    model_args.evaluate_during_training_steps = int(len(train_df) / train_batch_size) # after each epoch
    model_args.save_eval_checkpoints = False
    model_args.save_model_every_epoch = False
    model_args.wandb_project = colname
    model_args.train_batch_size = train_batch_size
    model_args.output_dir = save_dir
    model_args.best_model_dir = save_dir +"/best_model"
    model_args.cache_dir = save_dir + "/cache"
    model_args.tensorboard_dir = save_dir + "/tensorboard"
    model_args.regression = num_labels == 1
    model_args.gradient_accumulation_steps = gradient_accumulation_steps
    model_args.wandb_kwargs = {"reinit": True}
    model_args.fp16 = False
    model_args.fp16_opt_level = "O0"
    model_args.no_cache = False
    model_args.no_save = cross_validate
    model_args.save_optimizer_and_scheduler = True

    model = ClassificationModel(model.split("-")[0], model,
                                use_cuda=torch.cuda.is_available(),
                                num_labels=num_labels,
                                args=model_args)

    model.train_model(train_df,
                      eval_df=eval_df,
                      accuracy=accuracy,
                      precision=precision,
                      recall=recall,
                      f1=f1,
                      args={"use_multiprocessing": False,
                            "process_count": 1,
                            "use_multiprocessing_for_evaluation": False},)
    return model

def predict(fname, model_path, model=None,
            model_type="roberta-base", predict_list=None,
          index_list=None, index_colname="index"):

    print(model_path)

    if model is None:
        model = ClassificationModel(
            model_type.split("-")[0], model_path
        )

    preds, outputs = model.predict(predict_list)
    with open(model_path + '/' + fname + '_preds.txt', 'w') as f:
        f.write(f"{index_colname}\tpred\n")
        for index, pred in zip(index_list, preds):
            f.write(f"{index}\t{pred}\n")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train", action='store_true',
                        help="If true, train model.")
    parser.add_argument("--train_data", type=str,
                        default="data/paired_annotations.csv",
                        help="Input csv file.")
    parser.add_argument("--dev_split_size", type=float, default=0,
                        help="Percentage of data to hold out for validation.")
    parser.add_argument("--num_train_epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--text_cols", type=str, help="Text columns, comma separated.")
    parser.add_argument("--label_col", type=str, help="Column to evaluate.")
    parser.add_argument("--balance_labels", action='store_true',
                        help="If true, balance label distributions via equal sampling.")

    parser.add_argument("--cv", action='store_true',
                        help="If true, run cross validation.")


    parser.add_argument("--predict", action='store_true',
                        help="If true, predict.")
    parser.add_argument("--predict_data", type=str,
                        default="data/paired_utterances.csv",
                        help="Input csv file.")
    parser.add_argument("--predict_index_col", type=str,
                        help="Index column for mapping predictions to input.")

    parser.add_argument("--model_type", type=str, default="roberta-base",
                        help="Model type.")
    parser.add_argument("--output_dir", type=str, default="outputs/roberta",
                        help="Output directory.")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    print("Loading data from %s" % args.train_data)
    train_data = pd.read_csv(args.train_data).sample(frac=1)
    train_data = train_data[~train_data[args.label_col].isnull()]
    print("Loaded %d training examples." % len(train_data))

    model_type = args.model_type
    text_cols = args.text_cols.split(",")
    print(text_cols)
    output_dir = args.output_dir
    model = None

    if args.cv or args.train:
        print("Using %s as label" % args.label_col)

        if len(text_cols) == 1:
            train_data = train_data.rename(columns={text_cols[0]:
                                                        'text',
                                                    args.label_col: 'labels'})
            if args.train:
                cols = ["text", "labels"]
            else:
                cols = [args.predict_index_col, "text", "labels"]

        elif len(text_cols) == 2:
            train_data = train_data.rename(columns={text_cols[0]: 'text_a',
                                                    text_cols[1]: 'text_b',
                                                    args.label_col: 'labels'})
            if args.train:
                cols = ["text_a", "text_b", "labels"]
            else:
                cols = [args.predict_index_col, "text_a", "text_b", "labels"]
        else:
            print("You can have up to 2 texts to classify!")
            exit()
        train_data = train_data[cols].dropna()

    if args.train:

        if args.dev_split_size > 0:
            train_df, eval_df = train_test_split(train_data, test_size=0.2)
        else:
            train_df = train_data
            eval_df = train_data
        model = train(args.label_col,
                      train_df,
                      eval_df,
                      text_cols,
                      output_dir,
                      model_type,
              num_train_epochs=args.num_train_epochs,
                      balance_labels=args.balance_labels)
    if args.cv:

        n = 5
        kf = KFold(n_splits=n, random_state=42, shuffle=True)
        k = 0


        for train_index, val_index in kf.split(train_data):
            print("Split %d" % k)
            output_dir_k = output_dir + "/" + args.label_col + "_k%d" % k

            train_df = train_data.iloc[train_index]
            eval_df = train_data.iloc[val_index]
            model = train(args.label_col, train_df, eval_df, text_cols, output_dir=output_dir_k,
                          model=model_type, num_train_epochs=args.num_train_epochs, balance_labels=args.balance_labels,
                          cross_validate=True)

            if len(text_cols) == 1:
                predict_list = eval_df["text"].tolist()
            elif len(text_cols) == 2:
                predict_list = eval_df[["text_a", "text_b"]].values.tolist()
            else:
                print("You can have up to 2 texts to classify!")
                exit()
            index_list = eval_df[args.predict_index_col].tolist()
            fname = args.label_col + "_" + args.train_data.split("/")[-1].split(".")[0] + "_split_%d" % k
            predict(fname, output_dir_k, model, model_type, predict_list=predict_list,
                    index_list=index_list, index_colname=args.predict_index_col)


            k += 1

        if args.dev_split_size > 0:
            train_df, eval_df = train_test_split(train_data, test_size=0.2)
        else:
            train_df = train_data
            eval_df = train_data
        model = train(args.label_col,
                      train_df,
                      eval_df,
                      text_cols,
                      output_dir,
                      model_type,
                      num_train_epochs=args.num_train_epochs,
                      balance_labels=args.balance_labels)

    if args.predict:
        print("Loading data for prediction from %s" % args.predict_data)
        predict_data = pd.read_csv(args.predict_data)
        if len(text_cols) == 1:
            predict_df = predict_data.rename(columns={text_cols[0]: 'text'})[[args.predict_index_col,
                                                                              "text"]].dropna()
            predict_list = predict_df["text"].tolist()
        elif len(text_cols) == 2:
            predict_df = predict_data.rename(columns={text_cols[0]: 'text_a',
                                                      text_cols[1]: 'text_b',})[[args.predict_index_col,
                                                                                 "text_a", "text_b"]].dropna()
            predict_list = predict_df[["text_a", "text_b"]].values.tolist()
        else:
            print("You can have up to 2 texts to classify!")
            exit()
        index_list = predict_df[args.predict_index_col].tolist()
        fname = args.label_col + "_" + args.predict_data.split("/")[-1].split(".")[0]
        predict(fname, output_dir, model, model_type, predict_list=predict_list,
                index_list=index_list, index_colname=args.predict_index_col)