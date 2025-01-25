# American-Sign-Language-Image-Classification
## Overview
This project is machine learning classification model trained to identify images containing hand gestures in American Sign Language (ASL). It leverages a VGG-16 Convolutional Neural Network, a 16-layer architecture pretained on images from the ImageNet database. The performance of the VGG-16 model was compared 
to the Sequential Model from the Keras library, which we initially trained with the intention of exploring potential improvements as we gained insights into alternative architectures.
The [dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) for this project contained 29 folders: 26 corresponding to each letter in the English alphabet and 3 miscellaneous classes labeled "SPACE", "DELETE" and "NOTHING". Since the test data only contained 29 images,
a larger portion from the original 87,000 images got allocated for validation and testing to achieve a more robust result. In this repository, you will have access to the Python Notebook and documentation file containing results.  

The motivation for this initiative stems from research indicating that:
> "about 3.6% of the U.S. population, or about 11 million individuals, consider themselves deaf or have serious difficulty hearing"
>
>  — *[American Community Survey (ACS)](https://nationaldeafcenter.org/faq/how-many-deaf-people-live-in-the-united-states/), 2021*

> "there are more than 300 different sign languages in the world, spoken by more than 72 million deaf or hard-of-hearing people worldwide"
>
>  — *[National Geographic](https://education.nationalgeographic.org/resource/sign-language/), 2024*

With such a significant fraction of the population relying on ASL and other sign languages as a form of communication, this project aims to bridge the gap enhancing integration and accessibility in our online world.

## Outline of Process
**Architectures**: Sequential Model, VGG-16

**Technologies/Libraries**: Python, Tensorflow, Keras, Pandas, NumPy, Matplotlib, Scikit-learn

### Sequential Model
**1. Data Preprocessing**

As the dataset came packaged in separate folders and that Google Colab was the development environment, all of the folders were uploaded to Google Drive. We used 
the Drive Mount feature to connect the Colab Notebook to Google Drive and access the folders through there. Interestingly, the miscellaneous classes—"SPACE", "DELETE", and "NOTHING"—
included naturally occurring variability and imperfections. These classes provided a valuable opportunity for the model to learn to distinguish between relevant hand gestures and irrelevant or non-gesture signals, helping to prevent overfitting to perfect examples and promoting robustness. 

**2. Model Selection**

Prior to learning about the VGG-16 model, the Sequential Model was most familiar at the time for image classification use cases like this.

We used many layers including:

**Conv2D**: For extracting features

**MaxPooling2D**: To change the dimensions for more efficient processing

**Flatten**: Flatten the feature mappings to 1 dimension

**Dense**: Allow the model to learn from combinations of features

**Dropout**: Reduce overfitting by preventing overreliance on any features

**3. Training and Evaluation**

- The dataset was split using a 70-15-15 (70% for training, 15% for validation, 15% for testing)
- The images were resized to 200x200 pixels to ensure suitability for the model
- We incorporated multiple metrics from sklearn including Recall, Precision, and F1 Scores to statistically analyze the performance of the model

For context on the evaluation methods:
- **Precision**: Measures the percentage of all the predictions the model said was true, how many are actually true 

  $True Positives / (True Positives + False Positives)$
  
- **Recall**: Measures the percentage of all the actual true predictions, how many were actually true 

  $True Positives / (True Positives + False Negatives)$

- **F1-Score**: A harmonic mean of precision and recall 

  $2 * (Precision * Recall) / (Precision + Recall)$

**4. Results**

![image](https://github.com/user-attachments/assets/1b71a472-0db2-4312-b6db-6d900f965a02)

![image](https://github.com/user-attachments/assets/4ef4fbec-577b-4515-986b-2b09fcff78c7)

This model performed very well, but we learned it was possible to do better, which led us to explore another model.

### VGG-16 Model
**1. Data Preprocessing**

This model went through the same process as the Sequential Model to maintain consistency. We mounted the drive again and sourced the images from each folder directory. 
To augment the dataset, we used the ImageDataGenerator from Keras to normalize pixel values on the range of [0, 1]. We modified the one-hot encoded labels and the dataframe to include the correct labels to work with the ImageDataGenerator. 

**2. Model Selection**

This model was selected after learning of its effectiveness for image classification problems. It is a pre-trained VGG-16 model that had been trained on the ImageNet dataset.


**3. Training and Evaluation**

- We loaded the VGG-16 model with the pretrained ImageNet weights
- Image size of 200x200 pixels
- Custom layers like Dense, Flatten, and Dropout were added to fine-tune the model by allowing it to learn complex combinations of features and further prevent overfitting
- The same sklearn metrics were used to understand the model's performance


**4. Results**

![image](https://github.com/user-attachments/assets/09cfdbef-ce81-4429-a4c6-c78095f4f6f1)

![image](https://github.com/user-attachments/assets/600eba2c-e1fe-40e9-a52f-0d0dadad9e87)

The VGG-16 Model ended up performing better than the Sequential Model!

## Challenges

- As this was a collaborative effort, downloading folders containing 87,000 images locally would have been very tedious to access across multiple machines, and accessing the notebook on the same page would have been more difficult as well. Google Colab has many useful ML packages that were
  helpful for this project and holding the data in shared folders was benefical as well. 
- Splitting the data into 2 portions like the standard 80/20 did not produce the best results, so we used a 70/15/15 split. We went with a 70/30 split at first, holding 30% temporarily before splitting that again to produce the validation and test sets. 
- Initially, we did not have the computing power to train a model on so many images, so we decided to use ~25% of all the data and split that so it would finish the training process faster. Of course, more data would provide stronger results
  so while it helped to reduce the number of images we ultimately decided to take advantage of the full dataset. Luckily, Google Colab has an option to upgrade the GPU, which we took advantage of. We ended up using the T4 GPU.

  
## Future Development
Considering the very recent transition to online meeting tools like Zoom and Microsoft Teams, we wanted to tackle a project that had real world application. There is room for development of a 3D CNN application that would take video input from sign language users and provide real-time translations.  
