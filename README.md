# llm-tts-factory: End-to-End LLM-Backbone TTS Training Framework

llm-tts-factory is a full suite of end-to-end training scripts designed for building llm backbone based TTS model from scratch. 

To begin with, taking inspiration from [Soprano](https://huggingface.co/ekwek/Soprano-1.1-80M), this repository allows you to train a Soprano style TTS model from the ground up. Because of its architecture, it features an **extra Decoder training step**. Instead of generating audio directly from discrete tokens, this model uses the **hidden states of the LLM** as inputs to generate high-fidelity audio. 

I think this can be nicely setup for anybody to train llm backbone TTS models from scratch. Can spend some time to make it more modular and user friendly. Would love to hear from interested people on this!
---

## Architecture

The framework is divided into three core stages:

### 1. Codec
- **Description:** Encodes raw audio into discrete units and decodes them back.
- **Current State:** A very naive codec encoder and decoder. There is a ton of scope for improvements here (e.g., swapping in RVQ, DAC, or EnCodec).

### 2. LLM Backbone
- **Model:** Qwen-based causal language model.
- **Description:** The core sequence-to-sequence autoregressive model. Takes text and predicts discrete audio representations, passing its hidden states to the decoder.

### 3. Decoder
- **Model:** Vocos-based decoder.
- **Description:** A dedicated vocoder trained with Multi-Resolution STFT and GAN losses to synthesize the final audio waveform directly from the LLM's continuous hidden states.
- **Training strategy:** Trained in multiple stages, first to nail reconstruction and then the perception quality using gan losses.

---

## Training & Inference Commands

You can train the entire pipeline from scratch using the following step-by-step commands.

### 1. Codec Stage
Train the audio codec encoder and decoder:
```bash
python codec_train.py
```

### 2. LLM Stage
Train the Qwen-based causal LLM to learn the mapping from text to audio representations:
```bash
python train_llm.py \
    --input-dir <input_dir> \
    --save-dir <save_dir> \
    --from-scratch
```

### 3. Decoder Stage
Freeze the LLM and train the Vocos decoder to reconstruct the audio from the LLM's hidden states:
```bash
python train_decoder.py \
    --input-dir <input_dir> \
    --save-dir <save_dir>
```

### 4. Inference
Run end-to-end inference using your trained LLM and Decoder pair to generate TTS:
```bash
python simple_inference.py \
    --text "hello, my name is soma siddhartha" \
    --llm-path <llm_path> \
    --decoder-path <decoder_path> \
    --out simple_inf_out.wav
```
