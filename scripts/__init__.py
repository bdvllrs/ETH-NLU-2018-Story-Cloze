from .scheduler import main as scheduler_main


def run(config, training_set, testing_set):
    print("Loading model", config.model)
    if config.model == "scheduler":
        scheduler_main(config, training_set, testing_set)
    else:
        raise Exception('Unknown model ' + str(config.model) + '.')
