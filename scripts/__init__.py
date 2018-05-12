from .scheduler import main as scheduler_main
from .vanilla_seq2seq import main as seq2seq_main
from .sentence_embedding import main as sentence_embedding_main


def run(config, training_set, testing_set):
    print("Loading model", config.model)
    if config.model == "scheduler":
        scheduler_main(config, training_set, testing_set)
    elif config.model == "seq2seq":
        seq2seq_main(config, training_set, testing_set)
    elif config.model == "sentence_embedding":
        sentence_embedding_main(config, training_set, testing_set)
    else:
        raise Exception('Unknown model ' + str(config.model) + '.')
