# FINETUNE VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech

Finetuning of [VITS](https://arxiv.org/abs/2106.06103): Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech.


## Pre-requisites
0. Python >= 3.6
0. Clone this repository
0. Install python requirements. Please refer [requirements.txt](requirements.txt)
    1. You may need to install espeak first: `apt-get install espeak`
0. Build Monotonic Alignment Search and run preprocessing if you use your own datasets.
```sh
# Cython-version Monotonoic Alignment Search
cd monotonic_align
python setup.py build_ext --inplace

```

## Get Meta MMS checkpoints
In order to finetune our models, we need to start with fully-trained VITS checkpoints.
We take advantage of Meta's work [MMS: Scaling Speech Technology to 1000+ languages](https://github.com/facebookresearch/fairseq/blob/main/examples/mms/README.md),
where checkpoints of multiple languages can be found.

1. Download the list of [iso codes](https://dl.fbaipublicfiles.com/mms/tts/all-tts-languages.html) of 1107 languages.
2. Find the iso code of the target language and download the checkpoint. Each folder contains 3 files: `G_100000.pth`,  `config.json`, `vocab.txt`. The `G_100000.pth` is the generator trained for 100K updates, `config.json` is the training config, `vocab.txt` is the vocabulary for the TTS model. 
```
# Examples:
wget https://dl.fbaipublicfiles.com/mms/tts/eng.tar.gz # English (eng)
wget https://dl.fbaipublicfiles.com/mms/tts/azj-script_latin.tar.gz # North Azerbaijani (azj-script_latin)
```
The above command downloads generator only, which is enough to run TTS inference. If you want the full model checkpoint which also includes the discriminator (`D_100000.pth`) and the optimizer states, download as follows.
```
# Example (full checkpoint: generator + discriminator + optimizer):
wget https://dl.fbaipublicfiles.com/mms/tts/full_model/eng.tar.gz # English (eng)

```

Alternatively use the following script:
```python download_MMS_ckpt.py```

## Data Processing
In order to launch the finetuning script, we frist need to process the data.


To train on a novel identity we need to provide a new file called ```$VOICE_ID.txt``` in ```filelists```.
Each line in the file should have the following information:

```
{path to wav file} | {transcript of the audio file}
```
Then we need to split the data in test, train and val splits.

If list of all audio files and a list of all trascript files is available, we can directly run the following script:
```
python process_data.py --datapath $PATH_TO_DATAFOLDER --dataset_name $DATASET_NAME --lang $TARGET_LANG
```
where:
  - PATH_TO_DATAFOLDER: is the path of the folder containing all processed data.
  - DATASET_NAME: is the name of the folder containing the person specific data.
  - TARGET_LANG: is the iso code for the language we are training over.

Once the data is processed, we need to provide a new config file called ```$VOICE_ID.json``` in ```configs```:

```
{
    "train": {...}
      
    "data": {
      "training_files":"filelists/new_identity_train_filelist.txt",
      "validation_files":"filelists/new_identity_val_filelist.txt",
      "text_cleaners":[
          "transliteration_cleaners"
      ],
      "lang": $TARGET_LANG,
      "max_wav_value": 1.0,
      "sampling_rate": 16000,
      "filter_length": 1024,
      "hop_length": 256,
      "win_length": 1024,
      "n_mel_channels": 80,
      "mel_fmin": 0.0,
      "mel_fmax": null,
      "add_blank": true,
      "n_speakers": 0,
      "cleaned_text": true
    },
    "model": {...},
    "finetune": {
      "ckpt_path": $PATH_TO_MMS_CHECKPOINT,
      "epochs": 10000,
      "learning_rate": 2e-4
    }
  }
```

where:
  - TARGET_LANG: is the iso code for the language we are training over
  - PATH_TO_MMS_CHECKPOINT: is the path to the downloaded MMS checkpoints


## Finetuning
Now we have everything ready to start finetuning VITS over our own voices.
To start the process you can run:
```
python finetune.py -c configs/$VOICE_ID.json -m $VOICE_ID
```

## Inference
We can now run inference using our new finetuned model.
To run inference you can call:
```
python inference.py -m $VOICE_ID
```
We can finally write the text we want our voice to pronounce!