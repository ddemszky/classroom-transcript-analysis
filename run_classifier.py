"""
python supervised_classification.py

To run prediction:
python supervised_classification.py --predict

To run cross-validation:
python supervised_classification.py --cv

Specify column:
python supervised_classification.py --cv --col COLNAME
"""
from simpletransformers.classification import ClassificationModel
from argparse import ArgumentParser
from sklearn.model_selection import KFold, train_test_split
from scipy.stats import pearsonr, spearmanr
import warnings
import pandas as pd
from sys import exit
warnings.filterwarnings("ignore")


def pearson_corr(preds, labels):
    return pearsonr(preds, labels)[0]

def spearman_corr(preds, labels):
    return spearmanr(preds, labels)[0]

def train(colname, train_df, eval_df, text_cols,
          output_dir, model="roberta", num_labels=2,
          num_train_epochs=10,
          train_batch_size=4, gradient_accumulation_steps=4,
          max_seq_length=512,
          cross_validate=False):
    print("Train size: %d" % len(train_df))
    print("Eval size: %d" % len(eval_df))

    train_args = {
        'reprocess_input_data': True,
        'overwrite_output_dir': True,
        'evaluate_during_training': True,  # change if needed
        'max_seq_length': int(max_seq_length / len(text_cols)),
        'num_train_epochs': num_train_epochs,
        'evaluate_during_training_steps': 500,
        'save_eval_checkpoints': False,
        'save_model_every_epoch': False,
        'wandb_project': colname,
        'train_batch_size': train_batch_size,
        'output_dir': output_dir + "/" + colname,
        'best_model_dir': output_dir + "/" + colname + "/best_model",
        'cache_dir': output_dir + "/" + colname + "/cache",
        'tensorboard_dir': output_dir + "/" + colname + '/tensorboard',
        'regression': True,
        'use_cuda': True,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'wandb_kwargs': {'reinit': True,},
        'fp16': False,
        'fp16_opt_level': 'O0',
        'no_cache': False,
        'no_save': cross_validate,
        'save_optimizer_and_scheduler': True,
    }

    model = ClassificationModel(model.split("-")[0], model, num_labels=num_labels, args=train_args)

    model.train_model(train_df, eval_df=eval_df, pearson_corr=pearson_corr, spearman_corr=spearman_corr)
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

    parser.add_argument("--cv", action='store_true',
                        help="If true, run cross validation.")
    parser.add_argument("--cv_folds", type=int, default=5,
                        help="Number of folds for cross validation.")

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

    print("Loading data from %s" % args.train_data)
    train_data = pd.read_csv(args.train_data)
    train_data = train_data[train_data[args.label_col].isnull()]
    print("Loaded %d training examples." % len(train_data))

    model_type = args.model_type
    text_cols = args.text_cols.split(",")
    output_dir = args.output_dir
    model = None

    if args.train:
        print("Using %s as label" % args.label_col)

        if len(text_cols) == 1:
            train_data = train_data.rename(columns={text_cols[0]:
                                                        'text',
                                                    args.label_col: 'labels'})[["text", "labels"]].dropna()
        if len(text_cols) == 2:
            train_data = train_data.rename(columns={text_cols[0]: 'text_a',
                                                      text_cols[1]: 'text_b',
                                                    args.label_col: 'labels'})[["text_a", "text_b",
                                                                                "labels"]].dropna()
        else:
            print("You can have up to 2 texts to classify!")
            exit()

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
              num_train_epochs=args.num_train_epochs)

    if args.predict:
        print("Loading data for prediction from %s" % args.input)
        predict_data = pd.read_csv(args.predict_data)
        if len(text_cols) == 1:
            predict_df = predict_data.rename(columns={text_cols[0]: 'text'})[[args.predict_index_col,
                                                                              "text"]].dropna()
            predict_list = predict_df["text"].tolist()
        if len(text_cols) == 2:
            predict_df = predict_data.rename(columns={text_cols[0]: 'text_a',
                                                      text_cols[1]: 'text_b',})[[args.predict_index_col,
                                                                                 "text_a", "text_b"]].dropna()
            predict_list = predict_df[["text_a", "text_b"]].tolist()
        else:
            print("You can have up to 2 texts to classify!")
            exit()
        index_list = predict_df[args.predict_index_col].tolist()
        fname = args.label_col + "_" + args.input.split("/")[-1].split(".")[0]
        predict(fname, output_dir, model, model_type, predict_list=predict_list,
                index_list=index_list, index_colname=args.predict_index_col)