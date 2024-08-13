# Sudoku Solver

This project is a comprehensive solution for solving Sudoku puzzles using computer vision and deep learning. The application processes an image of a Sudoku puzzle, recognizes the digits, solves the puzzle, and then overlays the solution back onto the original image.

## Features

- **Image Processing**: Uses OpenCV to detect the Sudoku grid, extract digits, and preprocess them for recognition.
- **Digit Recognition**: A Convolutional Neural Network (CNN) built with TensorFlow is used to recognize the digits from the preprocessed image.
- **Puzzle Solving**: Implements an exact cover algorithm to solve the Sudoku puzzle.
- **Solution Overlay**: The solved puzzle is overlaid back onto the original image, showing the solution in place.

## Project Structure

- **`app.py`**: Entry point for the application. Handles the workflow from loading the image to displaying the solved puzzle.
- **`processes.py`**: Contains functions for processing the image, including detecting the grid, extracting and cleaning the digits, and overlaying the solution.
- **`sudoku.py`**: Implements the Sudoku solver using an exact cover algorithm.
- **`process_helpers.py`**: Helper functions used in image processing, such as finding contours, drawing lines, and cleaning digit images.
- **`model.py`**: Defines the CNN model for digit recognition.
- **`model_wrapper.py`**: Handles training and loading the model, including a custom callback to stop training when desired accuracy is reached.
- **`preprocesses.py`**: Preprocessing functions for preparing the image, including grayscaling, thresholding, and noise removal.

## Setup and Installation

### Prerequisites

- Python 3.7 or above
- TensorFlow
- OpenCV

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/vraj-parmar/Sudoku-Solver.git
   cd Sudoku-Solver
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Training the Model

If you want to retrain the digit recognition model:

1. Place your training images in the `original/` folder.
2. Run the following command to start training:
   ```bash
   python model_wrapper.py --train --wts_path <path_to_save_weights>
   ```

### Running the Solver

To solve a Sudoku puzzle:

1. Place the Sudoku image in the root directory.
2. Run the application:
   ```bash
   python app.py --image <path_to_image>
   ```
3. The solved Sudoku puzzle will be displayed with the solution overlaid on the original image.

## How It Works

1. **Grid Detection**: The Sudoku grid is detected from the input image using contour detection.
2. **Digit Extraction**: Each cell of the grid is isolated, and the digit (if present) is extracted.
3. **Digit Recognition**: The extracted digits are resized and passed through the CNN model for classification.
4. **Puzzle Solving**: The recognized digits are used to form a Sudoku puzzle, which is then solved using a backtracking algorithm.
5. **Overlaying Solution**: The solution is mapped back to the image, and the solved puzzle is displayed.

## Acknowledgments

- The Sudoku solver algorithm is based on code from [this source](https://www.cs.mcgill.ca/~aassaf9/python/sudoku.txt) under the GNU General Public License.
- The project leverages TensorFlow for deep learning and OpenCV for image processing.
