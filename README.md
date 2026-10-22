# VoxAnon

Official code for the paper **[VoxAnon: A balanced dataset for a fair evaluation of speech anonymization](https://arxiv.org/abs/2610.13313)**.

This repository provides:

- **Speaker lists** for the training and test sets, in both balanced and unbalanced configurations, together with their speaker attributes
- **Utterance lists** for each set
- **Code and config files** to generate the different sets

> **Balanced configuration:** the data is available on Hugging Face at [OraneDufour/VoxAnon](https://huggingface.co/datasets/OraneDufour/VoxAnon), with the original and anonymized audio (B3, B4, B5).

---

## Repository structure

| Path | Content |
|------|---------|
| [`test/`](test/) | Speaker and utterance lists of the test sets used in the paper (balanced and unbalanced) |
| [`train/`](train/) | Speaker and utterance lists of the training sets used in the paper (balanced and unbalanced) |
| [`dataset_building/`](dataset_building/) | Scripts used to select the speakers of each set, with the config files used to generate the sets in the paper |
| [`dataset_building/speaker_selection/`](dataset_building/speaker_selection/) | Speaker metadata (see below) and selection scripts |

### Speaker metadata

The speaker lists and attributes used to generate the sets are in:

- [`VoxCeleb1_speakers_metadata_verified.csv`](dataset_building/speaker_selection/VoxCeleb1_speakers_metadata_verified.csv)
- [`VoxCeleb2_speakers_metadata_verified.csv`](dataset_building/speaker_selection/VoxCeleb2_speakers_metadata_verified.csv)

Only speakers whose **gender, age and accent have been verified** are included. The one exception is the VoxCeleb2 file: for some speakers the age could not be cross-checked, and this is flagged in the file.

---

## Installation

Only a few packages are needed, since the anonymizers and attackers live in separate repositories (see below).

```bash
pip install -r requirements.txt
```

---

## Anonymizers

The anonymizers are the baselines of the [2026 VoicePrivacy Challenge](https://hal.science/hal-05561895/):

| Name | Repository |
|------|------------|
| B3 | [DigitalPhonetics/speaker-anonymization](https://github.com/DigitalPhonetics/speaker-anonymization) |
| B4 | [m-pana/spk_anon_nac_lm](https://github.com/m-pana/spk_anon_nac_lm) |
| B5 | [deep-privacy/SA-toolkit](https://github.com/deep-privacy/SA-toolkit) |

---

## Attackers

Four attackers are evaluated in the paper:

| Attacker | Repository |
|----------|------------|
| ResNet-101 and RedimNet | [mrouvier/kiwano](https://github.com/mrouvier/kiwano/tree/main) |
| WavLM-ECAPA | [deep-privacy/sidekit](https://github.com/deep-privacy/sidekit/tree/main/egs/libri360_train) |
| Ecapa-SSL (VPC 2026 baseline) |https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2026|
