# Story Cloze

## Scheduler
The goal of this model is to get try to find the order of the story given shuffled sentences.
It gets 5 shuffled sentences and outputs for each sentences the probability that the sentence is in each position.

## Config
Config file is `config.json` file.

## Project architecture
- `models` folder of all model classes or functions
- `scripts` main files to run the model

## Add a model
1. Create a new file with all helper functions and model functions in the `models` folder.
2. Import in the `models/__init__.py` files that will need to be imported later for convenience.
3. Add a file in the `scripts` folder for the model main. It needs to take 3 parameters: config, training_set, testing_set.
4. In the `scripts/__init__.py` file in the `run` function definition, add after all elifs:
```python
elif config.model == "some_name":
    model_main(config, training_set, testing_set)
```
5. Change in the `config.json` file the model parameter for the new model name to try it.

### Model
- bi-directional LSTMS
- sum of all the sequences
- 2 conv2D, pool2D layers
- linear layer
- softmax layer

## Credits
- Adrien Benamira <[AdriBenben](https://github.com/AdriBenben)>
- Benjamin Devillers <[bdvllrs](https://github.com/bdvllrs)>
- Esteban Lanter <[elSomewhere](https://github.com/elSomewhere)>