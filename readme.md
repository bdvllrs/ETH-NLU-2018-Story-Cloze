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
The `utils.Config` class imports every json files from a given folder.
In this project, every json files in the `./config` folder.
To access a value just get as an object property (e.g. `config.batch_size` for the batch size).

## Edit the configuration
Do not change the `config/default.json` file, instead add a new file in the `config` folder.
The files are loaded in alphabetical order, and new value override previous value.

Some values can also be overridden by passing an argument when executing.
### Available args
- `--model [-m] slug of the model to use` 
- `--action [-a] action to use`
- `--nthreads [-t] number of threads`

## Project architecture
- `models` folder of all model classes or functions
- `scripts` main files to run the model

## Add a script
To create a new script, add a file in the `scripts` folder and add this code snippet:
```pyhon
from scripts import DefaultScript


class Script(DefaultScript):

    slug = 'script_name'

    def train(self):
        # Some code
        pass
        
    def test(self):
        # Some code
        pass
```

To use this script, change the configuration (via json files or argument) and change the value
of the `model` value to the value of the Script `slug` attribute. 

By default, the `action` config value is `train`. Tu use the `test` method, use the value `test`.

## Model list
- `scheduler`
- `seq2seq`
- `sentence_embedding`

## Credits
- Adrien Benamira <[AdriBenben](https://github.com/AdriBenben)>
- Benjamin Devillers <[bdvllrs](https://github.com/bdvllrs)>
- Esteban Lanter <[elSomewhere](https://github.com/elSomewhere)>
- [A Corpus and Cloze Evaluation for Deeper Understanding of Commonsense Stories](https://arxiv.org/abs/1604.01696), 2016 <br>
_Mostafazadeh, Nasrin  and  Chambers, Nathanael  and  He, Xiaodong  and  Parikh, Devi  and  Batra, Dhruv  and  Vanderwende, Lucy  and  Kohli, Pushmeet  and  Allen, James_ 