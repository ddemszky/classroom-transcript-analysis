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
import torch
import gc
import warnings
warnings.filterwarnings("ignore")


def pearson_corr(preds, labels):
    return pearsonr(preds, labels)[0]

def spearman_corr(preds, labels):
    return spearmanr(preds, labels)[0]

def train(colname, train_df, eval_df, text_col,
          output_dir, model="roberta", num_train_epochs=10, cross_validate=False):


    train_df = train_df.rename(columns={text_col: 'text', colname: 'labels'})[["text", "labels"]].dropna()
    eval_df = eval_df.rename(columns={text_col: 'text', colname: 'labels'})[["text", "labels"]].dropna()

    print(len(train_df))
    print(len(eval_df))
    print(train_df["labels"].isnull().sum())

    train_args = {
        'reprocess_input_data': True,
        'overwrite_output_dir': True,
        'evaluate_during_training': True,  # change if needed
        'max_seq_length': 512,
        'num_train_epochs': num_train_epochs,
        'evaluate_during_training_steps': 500,
        'save_eval_checkpoints': False,
        'save_model_every_epoch': False,
        'wandb_project': colname,
        'train_batch_size': 8,
        'output_dir': output_dir + "/" + colname,
        'best_model_dir': output_dir + "/" + colname + "/best_model",
        'cache_dir': output_dir + "/" + colname + "/cache",
        'tensorboard_dir': output_dir + "/" + colname + '/tensorboard',
        'regression': True,
        'use_cuda': True,
        'gradient_accumulation_steps': 2,
        'wandb_kwargs': {'reinit': True,},
        'fp16': False,
        'fp16_opt_level': 'O0',
        'no_cache': False,
        'no_save': cross_validate,
        'save_optimizer_and_scheduler': True,
    }

    model = ClassificationModel(model.split("-")[0], model, num_labels=1, args=train_args)

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
    parser.add_argument("--text_cols", type=str, help="Text columns.")
    parser.add_argument("--label_col", type=str, help="Column to evaluate.")
    parser.add_argument("--cv", action='store_true',
                        help="If true, run cross validation.")

    parser.add_argument("--predict", action='store_true',
                        help="If true, predict.")
    parser.add_argument("--predict_index_col", type=str,
                        help="Index column for mapping predictions to input.")

    parser.add_argument("--output_dir", type=str, default="logs/roberta",
                        help="Output directory.")
    parser.add_argument("--num_train_epochs", type=int, default=5,
                        help="Number of training epochs")
    args = parser.parse_args()

    print("Loading data from %s" % args.input)
    data = load_dataset(args.input, add_annotations=args.add_annotations,
                        usecols=[args.predict_index_col, args.essay_col],
                        )

    model_type = "roberta-base"
    text_col = args.essay_col
    output_dir = args.output_dir
    show_gpu('Initial GPU memory usage:')
    model = None

    if args.train:
        print(args.label_col)
        train_df = data
        eval_df = data
        model = train(args.label_col, train_df, eval_df, text_col, output_dir, model_type,
              num_train_epochs=args.num_train_epochs)
    if args.predict:
        predict_df = data.rename(columns={text_col: 'text'})[[args.predict_index_col, "text"]].dropna()
        predict_list = predict_df["text"].tolist()
        index_list = predict_df[args.predict_index_col].tolist()
        fname = args.label_col + "_" + args.input.split("/")[-1].split(".")[0]
        predict(fname, output_dir, model, model_type, predict_list=predict_list,
                index_list=index_list, index_colname=args.predict_index_col)


    if args.cv:
        n = 5
        kf = KFold(n_splits=n, random_state=42, shuffle=True)
        k = 0

        data = data[~data[args.label_col].isnull()]
        predict_df = data.rename(columns={text_col: 'text'})[[args.predict_index_col, "text"]].dropna()

        for train_index, val_index in kf.split(data):
            print("Split %d" % k)
            output_dir = args.output_dir + "_k%d" % k

            train_df = data.iloc[train_index]
            eval_df = data.iloc[val_index]
            model = train(args.label_col, train_df, eval_df, text_col, output_dir=output_dir,
                           model=model_type, num_train_epochs=args.num_train_epochs,
                          cross_validate=True)

            predict_df = eval_df.rename(columns={text_col: 'text'})[[args.predict_index_col, "text"]].dropna()
            predict_list = predict_df["text"].tolist()
            index_list = predict_df[args.predict_index_col].tolist()
            fname = args.label_col + "_" + args.input.split("/")[-1].split(".")[0] + "_split_%d" % k
            predict(fname, output_dir, model, model_type, predict_list=predict_list,
                    index_list=index_list, index_colname=args.predict_index_col)

            del model
            gc.collect()
            torch.cuda.empty_cache()
            k+=1
    if args.no_cv_train:
        train_df, eval_df = train_test_split(data, test_size=0.2)
        print(args.col)
        train(args.col, train_df, eval_df, text_col, output_dir, model_type,
                       num_train_epochs=args.num_train_epochs)