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

The preprocess function must take two arguments: 
- `word_to_index` which is a dict mapping words to index.
- `sentence` one sentence

##### Example
here is an example of a preprocess function. The goal is to set all sentences into tokens:

```python
def preprocess_fn(word_to_index, sentence):
    return list(map(lambda x: word_to_index[x] if x in word_to_index.keys() else word_to_index['<unk>'], sentence))
```

#### Changing the output of get
The output of the get method is controlled by the `output_fn` attribute, or by calling
`Dataloader.set_output_fn(callback)`.

The `output_fn` callback must take one argument:
- data a `utils.Data` object 
To get the batch out of the `Data` object, use `Data.batch`. The label can be accessed by `Data.label`
and the `Dataloader` instance by `Data.dataloader`.

### Vocabs
To compute the vocab of the Dataset, use `Dataloader.compute_vocab()`.
Save it with `Dataloader.save_vocab(file, size=-1)` that saves all vocab by default.
Load it from a file with `Dataloader.load_vocab(file, size=-1)`.

### Get a generator
```python
Dataloader.get_batch(batch_size, epochs, random=True)
```

## Add a script
To create a new script, add a file in the `scripts` folder and add this code snippet:
```python
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

You can use the attribute `self.config` which is automatically passed to the Script class.

## Config
The `utils.Config` class imports every json files from a given folder.
In this project, every json files in the `./config` folder.
To access a value just get as an object property (e.g. `config.batch_size` for the batch size).

### Checking if a config exists
You can use `config.is_set(key)` to check if the key is in the config.

Example:
```json
{
  "key1": {
    "key2": "value"
  }
}
```

To use the `key2` value, you can use `config.key1.key2`. To check if the `key2` value is set, 
you can do: `config.key1.is_set('key2')`.

### Edit the configuration
Do not change the `config/default.json` file, instead add a new file in the `config` folder.
The files are loaded in alphabetical order, and new value override previous value.

Some values can also be overridden by passing an argument when executing.
### Available args
- `--model [-m] slug of the model to use` 
- `--action [-a] action to use`
- `--nthreads [-t] number of threads`

### Available configuration
Here is the default template for the configuration:
```json
{
  "vocab_size": 20000,
  "embedding_size": 100,
  "hidden_size": 100,
  "batch_size": 64,
  "max_size": 25,
  "learning_rate": 0.01,
  "n_epochs": 10000,
  "save_model_every": 10,
  "test_every": 2,

  "model": "model_slug",
  "action": "train",

  "debug": true,

  "sent2vec": {
    "model": null,
    "embedding_size": 500
  },

  "sentiment_analysis": {
    "vocab_size": 5000,
    "max_length": 100
  }
}
```

## How to run model
You have to install github project : <[Infersent](https://github.com/facebookresearch/InferSent)>
Follow the instructions Dependencies & Download and set :
```json
{
"GLOVE_PATH":"/home/benamira/Bureau/InferSent/dataset/GloVe/glove.840B.300d.txt",
  "model_path":"/home/benamira/Bureau/InferSent/encoder/infersent.allnli.pickle",
}
```
to your own configuration in config.json file.

This project is based on pytorch.
This is mainly base on the publication <[UNSUPERVISED MACHINE TRANSLATION USING MONOLINGUAL CORPORA ONLY](https://arxiv.org/pdf/1711.00043.pdf)>
But main difference is that we use sentence embedding.

To plot losses (4 losses : 1/ all 2/autoencoder 3/cross 4/adversial), np arrays will be saved very 20 batches
and accuracy test np arrays will be saved every 5 batches at the end of an epoch (accuracy_test.npy).
An accuracy validation is also written every 60 batches on 20 batches
Every epoch, model will be save.

10 min for 20 batches GPU GTX.

```json
{
  "model": "concept_fb",
  "action": "train",
  "batch_size":32,
  "hidden_size": 300,
  "embedding_size": 4096, can change if you change the sentence embedding
  "n_layers":2,
  "attention_bolean": "True", attention mechanism activated or not
  "plot_every":20, losses plot
  "plot_every_test":5 accuracy plot
  "GLOVE_PATH":"/home/benamira/Bureau/InferSent/dataset/GloVe/glove.840B.300d.txt",
  "model_path":"/home/benamira/Bureau/InferSent/encoder/infersent.allnli.pickle",
}
```

TODO: Change way of calcul loss : cos->cos²->||(u-v)||=0
TODO: make modular adversial loss

## Credits
- Adrien Benamira <[AdriBenben](https://github.com/AdriBenben)>.
- Benjamin Devillers <[bdvllrs](https://github.com/bdvllrs)>.
- Esteban Lanter <[elSomewhere](https://github.com/elSomewhere)>.
- [A Corpus and Cloze Evaluation for Deeper Understanding of Commonsense Stories](https://arxiv.org/abs/1604.01696), 2016 <br>
_Mostafazadeh, Nasrin  and  Chambers, Nathanael  and  He, Xiaodong  and  Parikh, Devi  and  Batra, Dhruv  and  Vanderwende, Lucy  and  Kohli, Pushmeet  and  Allen, James_.
- [Unsupervised Learning of Sentence Embeddings using Compositional n-Gram Features NAACL](https://arxiv.org/abs/1703.02507),  2018<br>
    _Matteo Pagliardini, Prakhar Gupta, Martin Jaggi_.
- [A large annotated corpus for learning natural language inference](https://nlp.stanford.edu/pubs/snli_paper.pdf), 2015<br>
    _Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning_.
- [AllenNLP for Elmo Embeddings: Deep contextualized word representations](https://arxiv.org/abs/1802.05365), 2018<br>
    _Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, Luke Zettlemoyer_.
- [Jacob Zweig for Elmo embedding import code from Medium](https://towardsdatascience.com/elmo-embeddings-in-keras-with-tensorflow-hub-7eb6f0145440).
