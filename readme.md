# Story Cloze

## Dataloader
Loads Data for the story cloze test.
To use the dataloader, 
```python
from utils import Dataloader
```
The Dataloader class returns a `utils.Data` instance.
To simply get a `Data` object, use `Dataloader.get(index, amount)`.
### Callbacks
#### Pre-processing
One can control pre-processing by setting the `preprocess_fn` attribute or by calling
`Dataloader.set_preprocess_fn(callback)`.
#### Changing the output of get
The output of the get method is controlled by the `output_fn` attribute, or by calling
`Dataloader.set_output_fn(callback)`.
### Vocabs
To compute the vocab of the Dataset, use `Dataloader.compute_vocab()`.
Save it with `Dataloader.save_vocab(file, size=-1)` that saves all vocab by default.
Load it from a file with `Dataloader.load_vocab(file, size=-1)`.

### Get a generator
```python
Dataloader.get_batch(batch_size, epochs, random=True)
```

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

## Model list
- `scheduler`
- `seq2seq`
- `sentence_embedding`

## Credits
- Adrien Benamira <[AdriBenben](https://github.com/AdriBenben)>
- Benjamin Devillers <[bdvllrs](https://github.com/bdvllrs)>
- Esteban Lanter <[elSomewhere](https://github.com/elSomewhere)>