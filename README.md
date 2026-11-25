# Human Activity Recognition (HAR)

A deep learning project that recognizes six human activities from smartphone motion sensors using a custom Bidirectional LSTM and Temporal Attention model implemented in PyTorch.

## Activities Recognized
- Walking
- Walking Upstairs
- Walking Downstairs
- Sitting
- Standing
- Laying

## Model Architecture
- Custom RNN Cell
- Custom LSTM Cell
- LSTM Sequence Implementation
- Bidirectional LSTM
- Temporal Attention
- HARNet Classification Model

## Model Architecture and Temporal Attention

### Recurrent Neural Network Models

This project explores recurrent architectures for Human Activity Recognition using sequential smartphone sensor data. Each input window consists of 128 time steps and 9 sensor channels, allowing the models to learn temporal patterns associated with different human activities.

The implemented architectures include:

- **RNN:** A basic recurrent network used to learn sequential patterns from sensor measurements.
- **LSTM:** Introduces a memory mechanism to retain relevant information across time steps and reduce the limitations of conventional recurrent networks.
- **Bidirectional LSTM (BiLSTM):** Processes sequences in both forward and backward directions to capture contextual information from the entire sensor window.
- **HARNet with Temporal Attention:** Combines bidirectional LSTM representations with a temporal attention mechanism to assign different importance weights to individual time steps.

### Temporal Attention Mechanism

Instead of treating every time step equally, temporal attention learns which portions of the input sequence are more informative for activity classification.

The attention mechanism calculates a weight for each temporal step. These weights are normalized so that their sum equals one, allowing the network to construct a weighted representation of the sequence.

The resulting context vector emphasizes relevant temporal information before it is passed to the classification layer. This helps the model focus on meaningful movement patterns rather than relying equally on every sensor measurement.

### Model Design

The HARNet architecture uses the following processing pipeline:

1. Input sensor sequence of shape **128 × 9**.
2. Bidirectional LSTM for temporal feature extraction.
3. Temporal attention for selecting informative time steps.
4. Context vector generation using weighted sequence representations.
5. Fully connected classification layer.
6. Prediction of one of the six activity classes.

This design combines sequential learning with attention-based feature aggregation to improve activity recognition performance.
  
## Results
- **Test Accuracy:** 93.45%
- **Trainable Parameters:** 46,982
- **Dataset:** UCI Human Activity Recognition Using Smartphones

## Project Features
- Raw sensor time-series processing
- Subject-wise data splitting
- Custom recurrent layers
- Temporal attention mechanism
- Training and evaluation pipeline
- Confusion matrix and class-level evaluation
- Interactive Streamlit application

## Technologies
Python · PyTorch · NumPy · Pandas · Matplotlib · Streamlit

## Application Preview

The interactive application allows users to explore activity predictions, class probabilities, sensor signals, and temporal attention.

![HAR Streamlit Dashboard](streamlit_demo.png)

## Dataset

The project uses the UCI Human Activity Recognition Using Smartphones dataset. The dataset is not included in this repository and must be downloaded separately.

## Project Structure

- `src/` — model architecture, layers, training, evaluation, and preprocessing
- `app/` — Streamlit application
- `results/` — model, evaluation results, and visualization outputs
- `docs/` — project documentation

## Authors

Human Activity Recognition — Assignment 4
