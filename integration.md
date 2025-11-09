This project contains a trained handwritten digit recogniser. It was trained on the MNIST dataset, black blackground, white numbers, 28x28 images.

The original black and white (bilevel) images from NIST were size normalized to fit in a 20x20 pixel box while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.

The model is here: assets/models/mnist_digit_classifier.tres

Here is a sample from which you can find out how to prepare an image for inference: examples/mnist_drawing_classifier.gd