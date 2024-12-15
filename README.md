# EmoTag: Emotion Analysis in Lyrics and Poetry

You will need GloVe to run the tests, download it from here: [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/).

Then place `glove.6B.200d.txt` in the main directory.

To run the tests:

```bash
python3 model_testing.py [ fnn | cnn | transformer | bert ]
```

E.g.

```bash
python3 model_testing.py transformer
```
