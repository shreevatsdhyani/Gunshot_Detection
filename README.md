# Gunshot Detection System with Hyperbolic Localization

This project implements a **Gunshot Detection System** capable of identifying the type of gunshot fired and determining its location using **hyperbolic localization**. The system utilizes audio signals captured from multiple microphones to detect the gunshot and calculate its position based on the time difference of arrival (TDOA).

## Features

- **Gunshot Type Detection**: Identifies the type of gunshot fired (e.g., AK-12, M16, etc.) based on audio features like MFCC (Mel-frequency cepstral coefficients).
- **Hyperbolic Localization**: Calculates the precise location of the gunshot based on the time differences of arrival (TDOA) at multiple microphones.
- **Real-time Audio Processing**: Uses libraries such as `librosa` for audio feature extraction and `cv2` for image-based processing of features.

## Requirements

Before running the system, ensure the following libraries are installed:

- Python 3.x
- `tensorflow` (for deep learning model)
- `librosa` (for audio processing)
- `numpy` (for numerical calculations)
- `opencv-python` (for image processing)
- `scikit-learn` (for machine learning tasks)
- `matplotlib` (for plotting)
  
You can install the required dependencies using:

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```text
tensorflow==2.9.0
librosa==0.9.1
numpy==1.21.2
opencv-python==4.5.3.56
scikit-learn==1.0.1
matplotlib==3.4.3
```

## Setup

### 1. Dataset Preparation

The project uses a dataset containing various gunshot sounds recorded from different guns. The dataset is stored in the following structure:

```
gunshots_audio/
  ├── AK-12/
  ├── AK-47/
  ├── IMI Desert Eagle/
  ├── M16/
  ├── M249/
  ├── M4/
  ├── MG-42/
  ├── MP5/
  └── Zastava M92/
```

Each folder corresponds to a specific gun type, and contains `.wav` audio files of gunshot sounds.

Make sure the dataset is placed in the correct path, and update the `data_dir` variable in the code to point to your dataset directory.

### 2. Feature Extraction

The system extracts **MFCC features** from audio files to be used in training the classification model for gunshot detection.

### 3. Gunshot Detection Model

The model architecture is based on **MobileNet** or **VGG19**, which is fine-tuned for gunshot classification. The model is trained using the extracted MFCC features.

### 4. Hyperbolic Localization

For localization, multiple microphones are required. These microphones must be placed at known locations and synchronized to record the gunshot. The system uses the **time difference of arrival (TDOA)** between pairs of microphones to calculate the location of the gunshot using hyperbolic localization.

### 5. Running the System

To run the system, follow these steps:

1. **Train the Gunshot Detection Model**:
    - Ensure your audio dataset is properly set up and the paths are correct.
    - The `load_data` function will process the audio files, extract MFCC features, and train a model to classify gunshot types.

2. **Localize the Gunshot**:
    - Use the multiple microphone setup to capture the sound of the gunshot.
    - The system will calculate the TDOA and use hyperbolic localization to determine the position of the gunshot.

```bash
python gunshot_detection.py
```

### 6. Example

You can test the system on a sample gunshot sound by providing the path to an audio file:

```python
# Example usage for gunshot detection
file_path = "path/to/your/sample_audio.wav"
y, sr = librosa.load(file_path, sr=None)
features = extract_features(y)  # Function to extract MFCCs or mel spectrogram
predicted_gun = model.predict(features)
print(f"Detected Gunshot Type: {predicted_gun}")
```

### 7. Hyperbolic Localization Example

Here’s an example of how to localize the gunshot using multiple microphones:

```python
# Example usage for localization
microphones_positions = [(0, 0), (10, 0), (5, 5)]  # Example microphone coordinates
time_differences = [0.0, 0.5, 0.3]  # Example TDOA in seconds
gunshot_position = localize_gunshot(microphones_positions, time_differences)
print(f"Gunshot Position: {gunshot_position}")
```

## Hyperbolic Localization Algorithm

The localization part of the system works by solving a system of hyperbolic equations based on the time difference of arrival (TDOA) between microphones. The algorithm uses least squares fitting or optimization methods to find the intersection of the hyperbolas formed by each pair of microphones.

## Contributions

- **Gunshot Classification**: Uses a deep learning model trained on extracted MFCC features from audio files.
- **Localization**: Implements hyperbolic localization using TDOA from multiple microphones.

## Future Scope

- **Real-time Implementation**: The current system can be enhanced to run in real-time by integrating it with live microphone input.
- **Integration with IoT**: Connect the system to an IoT platform for real-time monitoring and alerts.
- **Advanced Localization Techniques**: Implement more advanced localization techniques, such as trilateration and machine learning-based localization.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

