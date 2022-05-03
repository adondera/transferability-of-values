import spacy
import re
from ekphrasis.classes.segmenter import Segmenter
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import emoji

nlp = spacy.load('en_core_web_lg')

text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
               'time', 'url', 'date', 'number'],
    # terms that will be annotated
    annotate={"hashtag", "allcaps", "elongated", "repeated",
              'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens

    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter="twitter",

    # corpus from which the word statistics are going to be used 
    # for spell correction
    corrector="twitter",

    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=True,  # spell correction for elongated words

    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,

    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)

tweet_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
               'time', 'url', 'date', 'number'],
    # terms that will be annotated
    annotate={},
    fix_html=True,  # fix HTML tokens

    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter="twitter",

    # corpus from which the word statistics are going to be used 
    # for spell correction
    corrector="twitter",

    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=True,  # spell correction for elongated words

    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,

    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)

# segmenter using the word statistics from Twitter
seg_tw = Segmenter(corpus="twitter")


# This preprocessing method was used for our experiments.
def cleaner5(tweet):
    tweet = emoji.demojize(tweet)
    return cleaner4(tweet)


def cleaner4(tweet):
    # remove pictures
    tweet = re.sub("pic.twitter.com/[A-Za-z0-9]+", "", tweet)

    # rectification for Sandy
    tweet = re.sub(" url ", "", tweet)
    tweet = re.sub(" at_user ", "", tweet)

    # remove numbers
    tweet = re.sub("[0-9]+", "", tweet)

    # deabbreviate most used abbreviations
    tweet = tweet.replace("#iuic", "#IsraelUnitedInChrist")
    tweet = tweet.replace("#tcot", "#TopConservativesOnTweeter")

    # custom preprocessor
    tweet = " ".join(tweet_processor.pre_process_doc(tweet))

    # remove tags
    tweet = re.sub("<[^\s]+>", "", tweet)
    tweet = tweet.replace("_", " ")

    # remove left usernames
    tweet = re.sub("@[^\s]+", "", tweet)

    # remove punctation
    tweet = " ".join([token.text for token in nlp(tweet) if not token.is_punct])

    # remove reserved words
    tweet = tweet.replace(" rt ", "")
    tweet = re.sub("^rt ", "", tweet)

    # manual word corrections
    tweet = tweet.replace(" s ", " is ").replace(" al ", " all ").replace(" nt ", " not ").replace(" ppl ",
                                                                                                   " people ").replace(
        " m ", " am ").replace(" u ", " you ").replace(" r ", " are ").replace(" w ", " with ")

    # remove math signs
    tweet = tweet.replace("+", "").replace("=", "").replace(">", "").replace("<", "").replace("|", "")
    tweet = tweet.replace("https", "").replace("http", "")

    # manual ALM and BLM word splitting
    tweet = tweet.replace(" alllivesmatter ", " all lives matter ").replace(" alm ", " all lives matter ")
    tweet = tweet.replace(" blacklivesmatter ", " black lives matter ").replace(" blm ", " black lives matter ")

    # remove extra white spaces
    tweet = " ".join(tweet.split())
    return tweet.lower()


def cleaner3(tweet):
    tweet = tweet.lower()
    tweet = re.sub("^rt ", "", tweet)
    tweet = re.sub("pic.twitter.com/[A-Za-z0-9]+", "", tweet)
    tweet = " ".join(text_processor.pre_process_doc(tweet))
    tweet = re.sub("<[^\s]+>", "", tweet)
    tweet = tweet.strip()
    tweet = " ".join(tweet.split())
    tweet = " ".join([token.lemma_ for token in nlp(tweet) if not token.is_punct and not token.is_stop])
    return tweet.lower()


def cleaner2(tweet):
    tweet = tweet.lstrip('\"')
    tweet = tweet.rstrip('\"')
    tweet = remove_emojis(tweet)
    tweet = tweet.lower()
    tweet = re.sub("^rt", "", tweet)
    tweet = re.sub("\s[0-9]+\s", "", tweet)

    # remove usernames
    tweet = re.sub("@[^\s]+", "", tweet)
    tweet = re.sub("at_user", "", tweet)

    # remove # sign 
    tweet = tweet.replace("#", "").replace("_", " ")

    # remove urls
    tweet = re.sub("pic.twitter.com/[A-Za-z0-9]+", "", tweet)
    tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet)
    tweet = tweet.replace("url", "")

    tweet = tweet.strip()
    tweet = " ".join(tweet.split())

    # removes stop words
    tweet = " ".join([token.lemma_ for token in nlp(tweet) if not token.is_punct])

    return tweet


def cleaner1(tweet):
    # remove usernames
    # tweet = re.sub("@[A-Za-z0-9]+","",tweet)
    tweet = remove_emojis(tweet)
    tweet = tweet.lower()
    tweet = re.sub("^rt", "", tweet)
    tweet = re.sub("\s[0-9]+\s", "", tweet)

    # remove usernames
    tweet = re.sub("@[^\s]+", "", tweet)
    tweet = re.sub("at_user", "", tweet)

    # remove # sign 
    tweet = tweet.replace("#", "").replace("_", " ")

    # remove urls
    tweet = re.sub("pic.twitter.com/[A-Za-z0-9]+", "", tweet)
    tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet)
    tweet = tweet.replace("url", "")

    tweet = tweet.strip()
    tweet = " ".join(tweet.split())

    # removes stop words
    tweet = " ".join([token.lemma_ for token in nlp(tweet)])

    return tweet


def remove_emojis(data):
    emoj = re.compile("["
                      u"\U0001F600-\U0001F64F"  # emoticons
                      u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                      u"\U0001F680-\U0001F6FF"  # transport & map symbols
                      u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                      u"\U00002500-\U00002BEF"  # chinese char
                      u"\U00002702-\U000027B0"
                      u"\U00002702-\U000027B0"
                      u"\U000024C2-\U0001F251"
                      u"\U0001f926-\U0001f937"
                      u"\U00010000-\U0010ffff"
                      u"\u2640-\u2642"
                      u"\u2600-\u2B55"
                      u"\u200d"
                      u"\u23cf"
                      u"\u23e9"
                      u"\u231a"
                      u"\ufe0f"  # dingbats
                      u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)
