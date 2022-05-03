import argparse
import warnings
from nlp.transferability.bert_transferability import evaluate_bert
from nlp.transferability.lstm_transferability import evaluate_LSTM
from nlp.transferability.fasttext_transferability import evaluate_fasttext

# Example of running an experiment:
# python3 transferability_script.py --model=bert --target-domain="ALM" --fine-tune=True --target-frac=0.9

function_dict = {
    'bert': evaluate_bert,
    'fasttext': evaluate_fasttext,
    'lstm': evaluate_LSTM,
}

warnings.filterwarnings("ignore")

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='bert')
parser.add_argument('--target-domain', type=str)
parser.add_argument('--fold', type=bool, default=True)
parser.add_argument('--foundations', type=bool, default=False)
parser.add_argument('--size', type=int, default=35000)
parser.add_argument('--fine-tune', type=bool, default=False)
parser.add_argument('--train-all', type=bool, default=False)
parser.add_argument('--target-frac', type=float, default=1)
parser.add_argument('--target-experiment', type=bool, default=False)
parser.add_argument('--source-domain', action='append', default=[])

args = parser.parse_args()

evaluate_function = function_dict[args.model]

print(f"Evaluating transferability with the target domain: {args.target_domain}")
evaluate_function(args.target_domain, source_domains=args.source_domain, do_kfold=args.fold,
                  use_foundations=args.foundations,
                  dataset_size=args.size, fine_tune=args.fine_tune, train_all=args.train_all,
                  target_frac=args.target_frac, target_experiment=args.target_experiment)
