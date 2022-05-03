import datetime
from sklearn.metrics import classification_report


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round(elapsed))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def classification(y_true, y_predicted, output_dict=False, target_names=None):
    # in case of a single target name add it's counterpart to the list
    if target_names is not None and len(target_names) == 1:
        label = target_names[0]
        target_names = [f'non-{label}', label]

    return classification_report(y_true, y_predicted, output_dict=output_dict, target_names=target_names)
