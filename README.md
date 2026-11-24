# Image colorization with a U-Net decoder on a pretrained VGG16 encoder

Predicts the colours of a grayscale photo. Each image is converted to the CIELAB colour space; the network
sees only the lightness channel *L* and predicts the two colour channels *a* and *b*, which are then
recombined with *L* to give an RGB image.

## Model

- **Input:** the *L* channel at 256×256, scaled to [0, 1] and repeated to three channels for VGG16.
- **Encoder:** VGG16 pretrained on ImageNet, frozen (14.7 M parameters).
- **Skip connections** from `block1_conv2`, `block2_conv2`, `block3_conv3` and `block4_conv3`; the
  bottleneck is `block5_conv3` (16×16).
- **Decoder:** four stages of 2× upsampling, concatenation with the matching encoder features and two 3×3
  convolutions (512, 256, 128 and 64 channels); 10.2 M trainable parameters.
- **Output:** a 1×1 convolution with tanh gives *a* and *b* in [-1, 1].
- **Training:** mean squared error on *a, b*, Adam with learning rate 1e-4, mean absolute error tracked.

## Data

50,000 scene images from a Places365 subset on Kaggle (`bobaaayoung/place365`), resized to 256×256 and
split 90/10 into training and validation. The OpenCV preprocessing (RGB to Lab, normalisation) runs inside a
`tf.data` pipeline with shuffling, batching (batch size 8) and prefetching.

## Running

Open `Automatic_Colourizer.ipynb` in Google Colab with a GPU runtime. The first cells ask for your own
`kaggle.json` API token and download the dataset (about 680 MB). The notebook then shows the Lab channels of a
sample image, builds the input pipeline and the model, and trains it.

TensorFlow 2 (Keras), OpenCV, NumPy and Matplotlib are all preinstalled on Colab.

## Status

The saved notebook shows the start of a 50-epoch run (about 5,600 steps and 30 minutes per epoch on Colab).
Trained weights and colorized examples are not included yet.
