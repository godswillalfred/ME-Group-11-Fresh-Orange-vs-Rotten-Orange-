# Brief Project Report

Group ME11 developed a binary image classifier for distinguishing fresh oranges
from rotten oranges. Images were obtained from the Fruits Fresh and Rotten for
Classification dataset on Kaggle. The orange classes were selected, checked for
invalid and duplicate files, and divided into training, validation and test
sets. A MobileNetV3Small transfer-learning model was trained and evaluated in
Google Colab before being connected to a Streamlit web application. To use the
application, the user uploads a clear orange image, presses **Analyse Orange**,
and receives the predicted condition with a confidence score. The main
challenges were variations in lighting, backgrounds, stages of decay and the
binary model's tendency to classify unsupported images. Data augmentation,
early stopping, threshold selection and an uncertainty warning reduced these
problems. Future improvement should use more locally collected orange images
and a separate validator that rejects images containing no orange.

**Word count:** 137  
**GitHub repository:** ADD_GITHUB_REPOSITORY_URL_HERE  
**Deployed application:** ADD_STREAMLIT_URL_HERE
