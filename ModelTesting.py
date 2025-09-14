"""
This script loads a trained cat image classification model and evaluates its performance on a separate test dataset. 
It loads test images, preprocesses them for MobileNetV2, and prints the test accuracy.
"""

from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.utils import image_dataset_from_directory # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # type: ignore

# Load the trained model
model = load_model('cat_identifier_model.h5')

# Parameters
batch_size = 16
img_height = 224
img_width = 224

# Load test dataset
test_ds = image_dataset_from_directory(
    "data_test",
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False,
    label_mode='int'
)

# Preprocess images exactly like MobileNetV2 expects
test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y))

# Evaluate
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc:.4f}")
