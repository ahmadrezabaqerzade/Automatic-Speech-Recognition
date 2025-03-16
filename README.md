<div align="center">
    <img src="https://github.com/ahmadrezabaqerzade/Automatic-Speech-Recognition/blob/main/images/speech-to-text-remixed.png" alt="Logo" width="" height="200">
  </a>

<h1 align="center">Automatic Speech Recognition</h1>
</div>



## 1. Problem Statement:

Speech, like images, text, and others, has certain characteristics that can be analyzed to extract semantic features and used in various domains.

Artificial intelligence has made significant advancements in the field of Audio and has achieved remarkable progress in understanding, analyzing, and working with Audio. Generally, artificial intelligence in speech encompasses several areas such as audio classification, speech recognition, speaker recognition, and more.

One of these domains is speech recognition, which is widely used in everyday life nowadays. Examples include virtual assistants like Alexa and Cortana, customer support services, translation, video subtitles, voice commands, and more. Simply put, speech recognition means converting speech into text. After extracting the semantic features of speech, it can also be transformed into text.In artificial intelligence, automatic speech recognition refers to machines and computers being able to analyze and decode speech based on learning, and then convert it into text.



https://github.com/ahmadrezabaqerzade/Automatic-Speech-Recognition/assets/94822419/634f69f6-1dfe-42cd-bf5f-4a68d11ca18f



**Many animals of even complex structure which live parasitically within others are wholly devoid of an alimentary cavity.**

## 2. Related Works

* **Transformers: Attention Is All You Need**

article: <a href = "https://arxiv.org/pdf/1706.03762v7.pdf" >link</a>

code: <a href = "https://github.com/huggingface/transformers">link</a>

1️⃣ Encoder: The encoder module in the Transformer model consists of a stack of identical layers. Each layer has two sub-modules: a multi-head self-attention mechanism and a position-wise feed-forward neural network. The self-attention mechanism allows the model to weigh the importance of different positions in the input sequence, while the feed-forward network applies non-linear transformations to each position separately.

2️⃣ Decoder: The decoder module also comprises a stack of identical layers. Similar to the encoder, each layer in the decoder has two sub-modules: a self-attention mechanism that prevents positions from attending to subsequent positions in the decoder, and an attention mechanism that allows the model to focus on relevant parts of the input sequence.

3️⃣ Self-Attention Mechanism: Self-attention enables the model to compute representations of each position in the input sequence by attending to other positions. It calculates attention weights for each position by considering similarities between all pairs of positions. This mechanism helps capture dependencies between different parts of the sequence.

4️⃣ Feed-Forward Neural Network: The feed-forward network applies a point-wise fully connected layer to each position in the encoder and decoder layers separately. It processes each position independently using a two-layer feed-forward neural network, introducing non-linearity to the model.

5️⃣ Positional Encoding: Since the Transformer model does not have explicit sequential information, positional encodings are added to the input embeddings to provide information about the order of the words in the sequence. This allows the model to learn the sequential relationships between different positions.

<img src = "https://github.com/ahmadrezabaqerzade/Automatic-Speech-Recognition/blob/main/images/transformer.png" weight = 300 height = 300>


* **Conformer: Convolution-augmented Transformer for Speech Recognition**

article: <a href = "https://arxiv.org/pdf/2005.08100v1.pdf" >link</a>

code: -

1️⃣ Convolutional Module: The Convolutional Module is added to the Conformer architecture to capture local dependencies and process the input speech features efficiently. It uses 2D convolutional layers to extract local contextual information from the input spectrogram, helping to capture important low-level acoustic patterns.

2️⃣ Transformer Encoder: The Transformer Encoder module in Conformer is similar to the one in the original Transformer model. It consists of multiple stacked layers, where each layer contains self-attention and feed-forward neural network sub-modules. The self-attention layer helps capture global dependencies, while the feed-forward network applies non-linear transformations to each position.

3️⃣ Dynamic Convolution: Dynamic Convolution is a key modification in the Conformer architecture. It introduces parameterized 1D depthwise separable convolutions, which adaptively adjust the receptive field size across layers. This allows the model to capture different lengths of context dependencies, making it more suitable for speech recognition tasks with varying-length input sequences.

4️⃣ Multi-Headed Self-Attention: Conformer employs multi-headed self-attention, similar to the original Transformer model. It allows the model to attend to different positions with different weights, enabling it to capture long-range dependencies and improve performance on speech recognition tasks.

5️⃣ Positional Encoding: Just like in the Transformer model, Conformer incorporates positional encoding to provide sequential information to the model. It helps the model understand the relative order and relationships between different elements in the input sequence.

<img src = "https://github.com/ahmadrezabaqerzade/Automatic-Speech-Recognition/blob/main/images/conformer.png" weight = 300 height = 300>

* **wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations**

article: <a href = "https://arxiv.org/pdf/2006.11477v3.pdf" >link</a>

official code: <a href = "https://github.com/facebookresearch/fairseq">link</a>

In terms of modules, the Wav2vec 2.0 framework consists of several key components:

1.Feature Encoder: The feature encoder processes the audio waveform using a stack of convolutional layers to extract low-level acoustic features.

2.Encoder: The encoder is built upon a variant of the Transformer model, which captures contextual dependencies in the extracted acoustic features. It consists of multiple layers of self-attention mechanisms and position-wise feed-forward networks.

3.Quantization: Wav2vec 2.0 introduces a quantization module that discretizes the outputs of the encoder, reducing the memory requirements during training.

4.Quantizer: The quantizer performs vector quantization to map the continuous embeddings from the encoder into a discrete codebook representation. This enables efficient computation and enables a contrastive loss to be applied during training.

5.Context Predictor: The context predictor is a Transformer-based decoder that takes the quantized representations as input and predicts the original context window from which the quantized vectors were derived. It helps to enforce context-aware learning and further improves the quality of learned representations.

<img src = "https://github.com/ahmadrezabaqerzade/Automatic-Speech-Recognition/blob/main/images/wav2vec.png" weight = 300 height = 300>

* **Whisper**

article: <a href = "https://arxiv.org/pdf/2005.01972v2.pdf" >link</a>

code: -

1.Preprocessing Module: This module involves initial data processing steps such as audio normalization, noise reduction, and feature extraction. It prepares the input audio for further analysis.

2.Acoustic Modeling Module: This module is responsible for mapping acoustic features to phonetic representations. It typically consists of recurrent neural networks (RNNs), convolutional neural networks (CNNs), or their combinations to capture temporal dependencies and frequency patterns in the input speech signals.

3.Language Modeling Module: The language modeling module focuses on predicting the probability of word sequences in a given context. It captures the statistical properties of the language to refine the recognition output by incorporating broader contextual information.

4.Alignment and CTC (Connectionist Temporal Classification) Module: The alignment module serves the purpose of frame-level alignment between the input speech and the corresponding text. The CTC module allows sequence-level training and decoding, handling variable-length inputs and outputs.

5.Transcription Module: This module is the final step of the ASR system. It converts the output of the alignment and CTC module into the recognized text or transcription.

<img src = "https://github.com/ahmadrezabaqerzade/Automatic-Speech-Recognition/blob/main/images/whisper.png" weight = 300 height = 300>

## 3. The Proposed Method

Since Transformers are widely used in speech recognition and many models are built based on Transformer architectures, we choose the base Transformer model to understand its structure and attempt to enhance it with new techniques.

## 4. Implementation
Below, you can see the relevant block diagrams for the training and inference sections:

* **train:**

<img src = "https://github.com/ahmadrezabaqerzade/Automatic-Speech-Recognition/blob/main/images/trainblock.png">

* **inference:**

<img src = "https://github.com/ahmadrezabaqerzade/Automatic-Speech-Recognition/blob/main/images/inferencestructure.png">



### 4.1. Dataset

**train.txt:** train audios id

**valid.txt:** valid audios id

**test.txt:** test audios id

On average, there are **19 tokens in a sound**, with a **minimum of 1 token** per word and a **maximum of 44 tokens**. Approximately **75%** of the sounds have **24 tokens or fewer**. The average number of repetitions for a token is **16.5 times**, with a **minimum of 1** repetition and a **maximum of 17,474** repetitions. In total, we have **14,416 tokens**, which can be increased to **14,420 by adding special tokens**. We have **6 tokens** in the vocabulary that were repeated more than **5,000 times**, with the token **"the"** having the highest repetition count of **17,474**. After that, the tokens **",", ".", "of," and "and"** had the highest repetitions. **5,921 tokens** were repeated **only once**.You can see the histogram of repetition counts for the vocabulary tokens in this image:

<img src = "https://github.com/ahmadrezabaqerzade/Automatic-Speech-Recognition/blob/main/images/histogram.png">

The average duration of the sounds is **6.5 seconds**, with a **minimum sound duration** of **1.11 seconds**, and a **maximum** of **10.09 seconds**. **75%** of the sounds are **8.38 seconds** or shorter. The **average total** audio data is **0.000009**, with a **standard deviation** of **0.066**, calculated from the **audio data vectors**. 
 
### 4.2. Augment

**Add Noise:**






https://github.com/user-attachments/assets/96140626-a54c-403e-b346-7bb8a8980552










