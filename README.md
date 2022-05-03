# Cross-Domain Classification of Moral Values

We perform the first comprehensive cross-domain evaluation of a value classifier.
To do so, we employ the [Moral Foundation Twitter Corpus](https://journals.sagepub.com/doi/pdf/10.1177/1948550619876629),
consisting of seven datasets spanning different socio-political areas, annotated with the value taxonomy
of the [Moral Foundation Theory](https://cpb-us-e2.wpmucdn.com/sites.uci.edu/dist/1/863/files/2020/06/Graham-et-al-2013.AESP_.pdf).
We treat each dataset as a domain, and train three deep learning models in four settings,
with the goal of evaluating the generalizability, transferability, and catastrophic forgetting of value classifiers.

## :checkered_flag: Getting started
These instructions will allow you to get a copy of the project up and running on your local machine for testing purposes.
Additional details will be provided upon publication.

### :gear: Prerequisites
A step by step guide to get a development env running.

#### *Python*

Make sure you have the latest Python and Pip versions installed.

```
$ python --version
Python 3.8.5

$ pip --version
pip 20.2.4
```

We suggest to create a [virtual environment](https://docs.python.org/3/library/venv.html).
Install all packages. Please note that version conflicts may occur.

```
$ pip install -r requirements.txt --no-index
```

## :zap: Usage

Now you can run the experiments from the command line as in the following example:

```
python3 transferability_script.py --model=bert --target-domain="ALM" --fine-tune=True --target-frac=0.9

```

Refer to `transferability_script.py` for an overview of the parameters.


## :lock: License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
