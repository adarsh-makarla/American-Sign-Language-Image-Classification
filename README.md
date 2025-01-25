# American-Sign-Language-Image-Classification
## Overview
This project is machine learning classification model trained to identify images containing hand gestures in American Sign Language (ASL). It leverages a VGG-16 Convolutional Neural Network, a 16-layer architecture pretained on images from the ImageNet database. The performance of the VGG-16 model was compared 
to the Sequential Model from the Keras library, which we initially trained with the intention of exploring potential improvements as we gained insights into alternative architectures.
The [dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) for this project contained 29 folders: 26 corresponding to each letter in the English alphabet and 3 miscellaneous classes labeled "SPACE", "DELETE" and "NOTHING". Since the test data only contained 29 images,
a larger portion from the original 87,000 images got allocated for validation and testing to achieve a more robust result.

The motivation for this initiative stems from research indicating that:
> "about 3.6% of the U.S. population, or about 11 million individuals, consider themselves deaf or have serious difficulty hearing"
>
>  — *American Community Survey (ACS), 2021*

> "there are more than 300 different sign languages in the world, spoken by more than 72 million deaf or hard-of-hearing people worldwide"
>
>  — *National Geographic, 2024*

With such a significant fraction of the population relying on ASL and other sign languages as a form of communication, this project aims to bridge the gap enhancing integration and accessibility in our online world.

## Outline of Process
**Architectures**: Sequential Model, VGG-16

**Technologies/Libraries**: Python, Tensorflow, Keras, Pandas, NumPy, Matplotlib, Scikit-learn

### Sequential Model
**1. Data Preprocessing**

As the dataset came packaged in separate folders and that Google Colab was the development environment, all of the folders were uploaded to Google Drive. We used 
the Drive Mount feature to connect the Colab Notebook to Google drive and access the folders through there. Interestingly, the miscellaneous classes—"SPACE", "DELETE", and "NOTHING"—
included naturally occurring variability and imperfections. These classes provided a valuable opportunity for the model to learn to distinguish between relevant hand gestures and irrelevant or non-gesture signals, helping to prevent overfitting to perfect examples and promoting robustness. 

**2. Model Selection:**

Prior to learning about the VGG-16 model, the Sequential Model was most familiar at the time for image classification use cases like this.

We used many layers including:

**Conv2D**: For extracting features

**MaxPooling2D**: To change the dimensions for more efficient processing

**Flatten**: Flatten the feature mappings to 1 dimension

**Dense**: Allow the model to learn from combinations of features

**Dropout**: Reduce overfitting by preventing overreliance on any features

**3. Training and Evaluation**

The dataset was split using a 70-15-15 (70% for training, 15% for validation, 15% for testing). The images were resized to 200x200 pixels to ensure suitability for the model. We incorporated multiple metrics from sklearn including Recall, Precision, and F1 Scores to statistically analyze the performance of the model. 

**4. Results**

![image](https://github.com/user-attachments/assets/4ef4fbec-577b-4515-986b-2b09fcff78c7)


### VGG-16 Model
**1. Data Preprocessing**


**2. Model Selection:**


**3. Training and Evaluation**


**4. Results**




## Future Development
Considering the very recent transition to online meeting tools like Zoom and Microsoft Teams, we wanted to tackle a project that had real world application. There is room for development of a 3D CNN application that would take video input from sign language users and provide real-time translations.  
