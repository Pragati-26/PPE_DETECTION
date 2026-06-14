# PPE Detection

A Streamlit app for monitoring PPE compliance in workplace images using a custom YOLO model.

## Project Structure

- `app.py` - Streamlit interface for uploading images and running the PPE detection scan.
- `requirements.txt` - Python dependencies required to run the app.
- `models/best.pt` - Custom trained model weights used by the detector.
- `src/detector.py` - Backend detection logic.

## Setup

1. Create and activate a Python environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the model file exists at `models/best.pt`.

## Run

```bash
streamlit run app.py
```

## Usage

- Open the Streamlit app in your browser.
- Upload a workplace image in JPG or PNG format.
- Adjust confidence and IoU thresholds in the sidebar.
- Click `Run Safety Scan` to see the annotated image.

## Notes

- If the app fails to load the model, verify the model path in `app.py` and confirm `models/best.pt` is present.
- The app uses the `PPEDetector` class from `src/detector.py`.

deployed link : https://ppe-detection-bqjk.onrender.com/
