from .scheduler import main as scheduler_main
from .vanilla_seq2seq import main as seq2seq_main
from .sentence_embedding2 import main as sentence_embedding_main
from .sentiment_analysis import main as sentiment_analysis_main


def run(config, training_set, testing_set, sentiments):
    print("Loading model", config.model)
    if config.model == "scheduler":
        scheduler_main(config, training_set, testing_set)
    elif config.model == "seq2seq":
        seq2seq_main(config, training_set, testing_set)
    elif config.model == "sentence_embedding":
        sentence_embedding_main(config, training_set, testing_set)
    elif config.model == "sentiment_analysis":
        sentiment_analysis_main(config, training_set, testing_set, sentiments)
    else:
        raise Exception('Unknown model ' + str(config.model) + '.')
