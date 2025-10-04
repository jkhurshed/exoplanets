Clone repository or download it manualy
Run the following command to create python virtual environment:
    python3 -m venv myenv
Run to activate:
    source myenv/bin/activate
Then install the neccessary dependencies:
    pip install -r requirements.txt
To train and get saved model run:
    python exominer.py
This will give you trained and saved to .pkl and .keras. The .pkl file will contain everything needed to make new predictions, while the .keras file is the standard TensorFlow format that's most reliable for long-term storage.
When You Might Want the .keras File:
TensorFlow ecosystem - If you want to use TensorFlow tools like:

TensorFlow Serving

TensorFlow Lite (for mobile)

TensorFlow.js (for web)

Model sharing - If you want to share just the model architecture with others

Transfer learning - If you want to use this as a base for other models

Model visualization - Some tools work better with native Keras format

if you do not need .keras just ignore it.