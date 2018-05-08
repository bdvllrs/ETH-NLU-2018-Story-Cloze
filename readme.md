# Story Cloze

## Scheduler
The goal of this model is to get try to find the order of the story given shuffled sentences.
It gets 5 shuffled sentences and outputs for each sentences the probability that the sentence is in each position.

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