

### **Objective:**
Test a face recognition system for fairness (bias across demographics) and GDPR compliance using the CelebA and StyleGAN datasets.

---

## **Step 1: Set Up Your Environment**

### **A. Install Required Libraries**
Open a new Google Colab notebook and run the following commands to install the necessary libraries:

```python
!pip install opencv-python-headless
!pip install mtcnn
!pip install facenet-pytorch
!pip install torch torchvision
!pip install scikit-learn
!pip install matplotlib
```

### **B. Import Libraries**
Add the following imports to your notebook:

```python
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mtcnn import MTCNN
from facenet_pytorch import InceptionResnetV1
import torch
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
```

---

## **Step 2: Download and Prepare the CelebA Dataset**

### **A. Download the Dataset**
- Download the CelebA dataset from [Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset).
- Extract the ZIP file into a folder.

### **B. Upload to Google Colab**
- Upload the extracted folder to your Google Colab notebook.

---

## **Step 3: Load and Preprocess the CelebA Dataset**

### **A. Load the Dataset**
Create a function to load images and extract metadata from filenames:

```python
def load_celeba_dataset(dataset_path):
    images = []
    labels = []

    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg"):
            # Load image
            image_path = os.path.join(dataset_path, filename)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Append data
            images.append(image)
            labels.append(filename)  # Assuming filename contains label info

    return images, labels

# Example usage
dataset_path = "/content/CelebA"
images, labels = load_celeba_dataset(dataset_path)
```

### **B. Preprocess Images**
Resize and normalize images for consistency:

```python
def preprocess_images(images, target_size=(160, 160)):
    processed_images = []
    for image in images:
        image = cv2.resize(image, target_size)
        processed_images.append(image)
    return processed_images

processed_images = preprocess_images(images)
```

---

## **Step 4: Detect Faces Using MTCNN**

### **A. Initialize MTCNN**
```python
detector = MTCNN()
```

### **B. Detect Faces**
```python
def detect_faces(images, detector):
    detected_faces = []
    for image in images:
        faces = detector.detect_faces(image)
        if len(faces) > 0:
            # Extract the first face (assuming one face per image)
            x, y, w, h = faces[0]['box']
            face = image[y:y+h, x:x+w]
            detected_faces.append(face)
    return detected_faces

detected_faces = detect_faces(processed_images, detector)
```

---

## **Step 5: Extract Features Using FaceNet**

### **A. Initialize FaceNet**
```python
facenet_model = InceptionResnetV1(pretrained="vggface2").eval()
transform = transforms.Compose([transforms.ToTensor()])
```

### **B. Extract Embeddings**
```python
def extract_embeddings(faces, model, transform):
    embeddings = []
    for face in faces:
        face_tensor = transform(face).unsqueeze(0)
        embedding = model(face_tensor)
        embeddings.append(embedding.detach().numpy())
    return embeddings

embeddings = extract_embeddings(detected_faces, facenet_model, transform)
```

---

## **Step 6: Evaluate Fairness and Bias**

### **A. Group Embeddings by Demographic**
```python
def group_embeddings(embeddings, labels):
    # Group embeddings by gender or other attributes
    # Assuming labels contain demographic info
    male_embeddings = [embeddings[i] for i in range(len(embeddings)) if labels[i] == "male"]
    female_embeddings = [embeddings[i] for i in range(len(embeddings)) if labels[i] == "female"]

    return male_embeddings, female_embeddings

male_embeddings, female_embeddings = group_embeddings(embeddings, labels)
```

### **B. Calculate Metrics for Each Group**
```python
def calculate_similarity(embeddings):
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            similarities.append(similarity)
    return np.mean(similarities)

male_similarity = calculate_similarity(male_embeddings)
female_similarity = calculate_similarity(female_embeddings)

print(f"Average similarity for male faces: {male_similarity}")
print(f"Average similarity for female faces: {female_similarity}")
```

---

## **Step 7: Check for GDPR Compliance**

### **A. Data Minimization**
- Store only necessary data (e.g., face embeddings).
- Avoid storing raw images or personally identifiable information (PII).

### **B. Consent and Anonymization**
- Ensure the dataset includes proof of consent if using real faces.
- Use synthetic data if real data is unavailable or raises privacy concerns.

---

## **Step 8: Document Your Findings**
- Record the performance metrics for each demographic group.
- Highlight any disparities in accuracy or similarity scores.
- Document steps taken to ensure GDPR compliance.

---

---

### **Summary**
- **CelebA** is used for fairness and bias testing.
- **MTCNN** detects faces, and **FaceNet** extracts features.
- **Group embeddings by demographic** to evaluate fairness.
- **Calculate similarity scores** to identify bias.
- **Ensure GDPR compliance** by minimizing data and verifying consent.

