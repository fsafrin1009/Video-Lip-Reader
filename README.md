Video-Lip-Reader

The lip-reading project uses a deep learning model to predict the speech content from video frames. It utilizes a Conv3D-LSTM model to process the video frames and make predictions. The model is trained with a CTC (Connectionist Temporal Classification) loss function to deal with variable-length sequences.

Features:
1. Downloads a dataset (data.zip) containing videos and alignment files.
2. Loads the video frames and corresponding alignments from the dataset.
3. Prepares and preprocesses the data for training.
4. Builds a Conv3D-LSTM model for lip-reading.
5. Compiles the model with the CTC loss function and Adam optimizer.
6. Defines a scheduler and callback functions for model training.
7. Trains the model on the data and displays prediction results during training.
8. Provides a function to predict speech from a given video file.

Installation:
The required Python libraries can be installed using `!pip install opencv-python matplotlib imageio gdown tensorflow`.

Usage:
- Prepare the data and download the 'data.zip' file containing videos and alignment files.
- Build and compile the lip-reading model using `build_lip_reading_model()` and `lip_reading_model.compile()` functions.
- Train the model using the training data with `lip_reading_model.fit()`.
- Optionally, use `predict_on_video()` to predict speech from a video file.

Technologies/Frameworks:
- OpenCV: For video processing and frame extraction.
- TensorFlow: For building and training the deep learning model.
- Matplotlib: For displaying visualizations.
- ImageIO: For saving video frames as GIFs.
- Gdown: For downloading files from Google Drive.
