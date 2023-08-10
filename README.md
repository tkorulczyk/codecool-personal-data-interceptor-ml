# SDA NLP Neural Network Model

**SDA NLP Neural Network Model** is an advanced NLP (Natural Language Processing) model implemented in PyTorch. It was developed as a part of a specialized course at the Software Development Academy. The model processes text data, extracting features using TF-IDF and TruncatedSVD, and then feeds the resultant vectors to a neural network for classification.

<p align="center">
  <img src="project/img/data-protection.png?raw=true" alt="AI Generated Data protection" title="AI Generated Data protection" align="center">
</p>

**Disclaimer:** The image was generated using artificial intelligence and, as such, is not subject to copyright protection.

## About the Model

The NLP model is designed to accept text data, transform it into numerical representations using the TF-IDF Vectorizer, and then reduce the dimensionality using TruncatedSVD. This processed data is then used to train a neural network for predicting specific outcomes.

## Key Features

1. **Text Vectorization**: Converts text data into numerical vectors using the TF-IDF technique.
2. **Dimensionality Reduction**: Uses TruncatedSVD for efficient representation of text data.
3. **Neural Network Model**: A basic feed-forward neural network built with PyTorch for classification tasks.

## Implementation Details

- The `NLPNet` class defines the neural network architecture.
- `VectWordDataset` and `get_dataloader` help in processing the data for PyTorch's data loading utility.
- The training loop and evaluation metrics are defined in the `train_test_pt_model` function.

## Getting Started

1. **Setup Environment**: Ensure you have PyTorch, NumPy, Pandas, and scikit-learn installed in your Python environment.
2. **Clone the Repository**: Clone the source code repository to your local machine.
3. **Run the Model**: Navigate to the project's root directory and run the `main()` function from the provided script.

## Contributing

Contributions are welcome! If you'd like to improve the model or add new functionalities, please fork the repository, make your changes, and submit a pull request.

## License

This project is open-sourced under the MIT License.
