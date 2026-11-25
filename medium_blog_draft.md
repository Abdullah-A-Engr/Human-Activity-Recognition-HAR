# Can a Smartphone Recognize Human Activity from Motion Sensors?

Human Activity Recognition (HAR) is an interesting application of machine learning because it transforms simple motion measurements into meaningful information about human activities. A smartphone contains accelerometers and gyroscopes capable of recording movement, and these signals can be used to distinguish activities such as walking, climbing stairs, sitting, standing, and lying down.

For our Applied Machine Learning Assignment 4, we developed a Human Activity Recognition system using raw smartphone sensor signals and a recurrent neural network implemented in PyTorch. Rather than relying on precomputed features, our model processes time-series signals directly and learns temporal patterns from them.

Our final model achieved **93.45% accuracy on the test set**.

## 1. The Problem

Human movement produces different patterns in acceleration and rotational motion. Walking produces repeated motion patterns, stair movement introduces different dynamics, and stationary activities generally produce less variation.

However, recognizing activities from sensor signals is not always straightforward. Two activities may produce similar measurements, especially when the movement is limited. This makes Human Activity Recognition a useful time-series classification problem.

Our goal was to build a model capable of learning these patterns from smartphone motion data and classifying each sensor window into one of six activities:

- Walking
- Walking upstairs
- Walking downstairs
- Sitting
- Standing
- Laying

## 2. Working with Raw Sensor Data

We used the UCI Human Activity Recognition Using Smartphones dataset. The dataset contains recordings collected from volunteers carrying a smartphone around the waist while performing different activities.

Each sample contains 128 time steps recorded at 50 Hz, representing approximately 2.56 seconds of motion. We used nine sensor channels containing body acceleration, total acceleration, and gyroscope measurements along the three axes.

Instead of using the dataset's 561 precomputed features, we used raw inertial signals. This allowed the model to learn temporal patterns directly from the sensor sequence.

The dataset was separated by subjects for training, validation, and testing. This is important because windows from the same person may contain similar motion characteristics. A subject-based split provides a more meaningful evaluation of how the model performs on people it did not see during training.

Channel standardization statistics were calculated using training data only to avoid introducing information from validation or test data into preprocessing.

## 3. Implementing Recurrent Components

An important part of the assignment was implementing recurrent components rather than relying entirely on built-in recurrent layers.

We implemented a custom RNN cell, an LSTM cell using its gate equations, an LSTM sequence loop, and a bidirectional LSTM.

Recurrent neural networks are useful for sequential data because their hidden states carry information from previous time steps. This allows the network to learn how sensor patterns evolve over time rather than treating each measurement independently.

For our final architecture, we used a Bidirectional LSTM. It processes the sequence in both forward and backward directions, combining information from both directions.

Because our input is a complete recorded sensor window, the model can use information from both earlier and later positions within that window.

## 4. Adding Temporal Attention

After the Bidirectional LSTM, we implemented a temporal attention mechanism.

Rather than treating every time step equally, attention assigns a score to each hidden state. The scores are normalized into weights that sum to one, and those weights are used to produce a context vector.

The context vector is passed to the classification layer, which produces scores for the six activity classes.

This architecture gives us another way to inspect the model. We can visualize attention weights across a motion window and see how the model distributes its weighting over time.

Attention weights are useful for model inspection, but they should not be treated as definitive causal explanations of a prediction.

## 5. Training the Model

We trained the model for 30 epochs using the Adam optimizer.

Our training configuration was:

- Learning rate: 0.001
- Batch size: 64
- Gradient clipping norm: 1.0
- Training epochs: 30
- Trainable parameters: 46,982

The best checkpoint was selected using validation loss rather than simply using the final training epoch.

The minimum validation loss was **0.0765 at epoch 18**. Validation accuracy reached a peak of approximately **98.08% at epoch 3**. These values occurred at different epochs, which demonstrates why validation loss and validation accuracy should not be treated as interchangeable metrics.

![Training and validation loss curves](../results/figures/loss_curves.png)

The training curves show rapid learning during the early epochs. Later in training, validation loss begins increasing while training loss remains comparatively low. This is consistent with overfitting during later epochs, making checkpoint selection important.

## 6. Evaluating on Unseen Test Subjects

After training and selecting the checkpoint, we evaluated the model on the held-out test set containing 2,947 samples.

The final test accuracy was:

# 93.45%

The per-class accuracy was:

| Activity | Test Accuracy |
|---|---:|
| Walking | 97.58% |
| Walking upstairs | 99.58% |
| Walking downstairs | 98.81% |
| Sitting | 81.87% |
| Standing | 84.21% |
| Laying | 99.81% |

The results show strong performance for walking-related activities and laying. Sitting and standing were more difficult to distinguish.

![Confusion matrix](../results/figures/confusion_matrix.png)

The confusion matrix reveals that 87 sitting samples were classified as standing, while 84 standing samples were classified as sitting.

This makes sense from a sensor perspective. Both activities involve relatively limited movement, so their acceleration and gyroscope patterns can be similar. In comparison, walking-related activities contain more distinctive motion patterns.

The confusion between sitting and standing was the most significant limitation observed in our experiment.

## 7. Exploring Attention

We also generated attention visualizations for individual activity windows and average attention patterns.

These visualizations help us inspect how attention is distributed across the time steps of a sensor sequence. The model may assign different weights to different portions of a window, depending on the temporal patterns it learned.

![Average attention visualization](../results/figures/average_attention.png)

Attention visualizations provide an additional perspective beyond accuracy and the confusion matrix. However, a high attention weight does not necessarily mean that a particular time step independently caused the model's decision. It is best treated as a visualization of learned weighting.

## 8. Building an Interactive Demonstration

To make the model easier to explore, we also developed a Streamlit application.

The application allows users to select test windows and view predictions. It displays the predicted activity, probability scores for the six classes, sensor signal visualizations, and temporal attention weights.

This made it possible to interact with individual examples rather than only examining aggregate evaluation results.

The frontend was tested in Google Colab using an embedded Streamlit interface.

## 9. Limitations and Lessons Learned

The experiment demonstrated that raw motion signals can be used effectively for activity classification, but several limitations remain.

First, activities with similar sensor patterns can be difficult to distinguish. Sitting and standing were the clearest example in our results.

Second, sensor placement and orientation may affect signal characteristics. A model trained on one particular collection setup may not generalize equally to different phone positions.

Third, validation strategy matters. Randomly splitting highly related sensor windows may result in overly optimistic evaluation. Separating subjects provides a stronger assessment of generalization to unseen participants.

Finally, attention visualization is useful for inspecting a model, but it does not provide a complete explanation of the model's internal reasoning.

## 10. Conclusion

This project allowed us to implement and evaluate a complete time-series machine learning pipeline, from raw sensor preprocessing to model training, testing, visualization, and interactive inference.

Our Bidirectional LSTM with temporal attention achieved 93.45% test accuracy on the UCI Human Activity Recognition dataset.

The model performed particularly well on dynamic activities and laying, while sitting and standing remained more challenging.

Beyond the accuracy result, the project helped us understand recurrent neural networks, sequence processing, attention mechanisms, subject-based validation, overfitting, and the importance of evaluating models using data that was not used during training.

Human Activity Recognition is a good example of how machine learning can transform raw sensor measurements into useful predictions—and how examining model limitations is just as important as reporting its performance.
