import argparse
import warnings
from nlp.transferability.fasttext_transferability import evaluate_fasttext

warnings.filterwarnings("ignore")

datasets = ['ALM', 'Baltimore', 'BLM', 'Davidson', 'Election', 'MeToo', 'Sandy']
parser = argparse.ArgumentParser()
parser.add_argument('--fine-tune', type=bool, default=False)
parser.add_argument('--train-all', type=bool, default=False)
parser.add_argument('--foundations', type=bool, default=False)
parser.add_argument('--target-frac', type=float, default=1)
parser.add_argument('--target-experiment', type=bool, default=False)
args = parser.parse_args()

for domain in datasets:
    print(f"Evaluating transferability with the target domain: {domain}")
    evaluate_fasttext(domain, use_foundations=args.foundations, fine_tune=args.fine_tune,
                      train_all=args.train_all, target_frac=args.target_frac, target_experiment=args.target_experiment)
