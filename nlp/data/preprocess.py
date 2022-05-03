import sys
import pandas as pd
from tqdm import tqdm
import json
from nlp.data.cleaners import cleaner1, cleaner2, cleaner3, cleaner4, cleaner5

DUMMY_TEXT = "no tweet text available"

output = {"tweet_id": [], "text": [], "fairness": [], "non-moral": [], "purity": [], "degradation": [], "loyalty": [],
          "care": [], "cheating": [], "betrayal": [], "subversion": [], "authority": [], "harm": []}

moralValues = ["fairness", "non-moral", "purity", "degradation", "loyalty", "care", "cheating", "betrayal",
               "subversion", "authority", "harm"]


def getValues(selectedAnnotations):
    values = []
    for value in moralValues:
        values.append((value, int(value in selectedAnnotations)))
    return values


def annotationStrategy0(annotations):
    selectedAnnotations = ",".join([annotator['annotation'] for annotator in annotations])
    return getValues(selectedAnnotations)


def annotationStrategy1(annotations):
    majorityAnnotations = []
    addedValues = (",".join([annotator['annotation'] for annotator in annotations])).split(",")
    for value in moralValues:
        if addedValues.count(value) >= len(annotations) / 2:
            majorityAnnotations.append(value)
    if 'non-moral' in majorityAnnotations and len(majorityAnnotations) > 1:
        majorityAnnotations.remove('non-moral')
    selectedAnnotations = ",".join(majorityAnnotations)
    if selectedAnnotations == '':
        selectedAnnotations = 'non-moral'
    return getValues(selectedAnnotations)


def preprocessStrategy0(rawText):
    return rawText


def preprocessStrategy1(rawText):
    return cleaner1(rawText)


def preprocessStrategy2(rawText):
    return cleaner2(rawText)


def preprocessStrategy3(rawText):
    return cleaner3(rawText)


def preprocessStrategy4(rawText):
    return cleaner4(rawText)


# This is the strategy used in our experiments.
def preprocessStrategy5(rawText):
    return cleaner5(rawText)


preprocess = {"0": preprocessStrategy0, "1": preprocessStrategy1, "2": preprocessStrategy2, "3": preprocessStrategy3,
              "4": preprocessStrategy4, "5": preprocessStrategy5}


def jsonToCsv(output, corpusName):
    df = pd.DataFrame(output)
    return df.to_csv(f"./processed/{corpusName}_{strategy}.csv", index=False)


def mergeCorpuses(corpuses):
    combined_csv = pd.concat([pd.read_csv('./processed/' + f + f'_{strategy}.csv') for f in all_filenames])
    # export all to csv
    combined_csv.to_csv(f'./processed/MFTC_{strategy}.csv', index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    path = sys.argv[1]
    strategy = sys.argv[2]

    # Run python preprocess.py <path to MFTC_text.json> to preprocess all tweets

    with open(path) as rawJson:
        data = json.load(rawJson)
        for corpus in data:
            for key, _ in output.items():
                output[key] = []
            corpusName = corpus['Corpus']
            for tweet in tqdm(corpus['Tweets'], desc=f'Processing {corpusName:>{10}}'):
                if tweet['tweet_text'] == DUMMY_TEXT:
                    continue
                values = annotationStrategy1(tweet['annotations'])
                text = preprocess[strategy](tweet['tweet_text'])
                if text == "":
                    continue
                output['tweet_id'].append(tweet['tweet_id'])
                output['text'].append(text)
                for (value, flag) in values:
                    output[value].append(flag)
            jsonToCsv(output, corpus['Corpus'])
        all_filenames = [corpus['Corpus'] for corpus in data]
        mergeCorpuses(all_filenames)
