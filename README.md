



Group 2 - Leaf Guard

Intelligent
Plant Disease Detection System
Index
Problem Statement	4
Background Information	4
Applications	5
Motivation For Selection Of The Project	6
Dataset Description And Dataset Source	7
1. Overview	7
2. Purpose and Objectives	7
3. Data Composition	7
Class Examples	7
4. Image Characteristics	8
5. Data Collection Methodology	8
6. Augmentation and Preprocessing	9
7. Applications	9
8. Challenges and Considerations	9
9. How to Access and Use	10
Current Benchmark	10
Proposed Plan	10
Approach	11
Packages and Tools	11
Algorithms	11
Metrics	12
Stages And Deliverables	12
Deployment Plan	13
Key Components	13
Backend Development (FastAPI)	13
Frontend Development	14
Deployment on AWS	14
CI/CD Pipeline (GitHub Actions)	14
Configure Multilingual Support	14
Monitoring and Maintenance	14
Preliminary Exploratory Data Analysis	15
Expected Outcomes	17
Project Demonstration Strategy	18
Project Timeline	20
Leaf Guard Team	21


Problem Statement 
Plant diseases account for substantial agricultural losses globally, threatening food security and economic stability. While traditional disease detection relies on visual inspection by experts and farmers, this approach is limited by its dependence on human expertise, time-intensive nature, and potential inconsistencies. 
To address these challenges, we propose LeafGuard - an automated plant disease detection system powered by artificial intelligence and computer vision. 
This solution aims to provide rapid, accurate, and accessible disease identification, enabling early intervention and preventive measures. By transforming plant disease detection from a manual to an automated process, LeafGuard seeks to enhance crop protection, reduce agricultural losses, and contribute to global food security through timely and reliable disease diagnostics.
Background Information
Plant diseases silently devastate agricultural yields worldwide, posing a critical threat to both economic stability and global food security. Each year, these diseases claim a significant portion of crops, creating ripple effects that extend from local farming communities to national economies. While the impact is measurable in economic terms, the human cost—reflected in reduced food availability and compromised livelihoods—is profound.
Traditionally, the detection of plant diseases has relied heavily on human expertise. Plant pathologists and experienced farmers conduct visual inspections, drawing upon years of experience to identify and classify diseases. However, this conventional approach faces several critical limitations:
Time Constraints: Manual inspection of vast agricultural areas is time-consuming
Expertise Gap: There's a global shortage of qualified plant pathologists
Consistency Challenges: Human observations can vary based on fatigue and experience
Detection Delay: By the time symptoms are visible to the naked eye, diseases may have already spread significantly
The urgent need for more reliable and accessible disease detection methods has sparked innovation in agricultural technology. Modern automated systems, powered by artificial intelligence and computer vision, offer promising solutions for:
Rapid and accurate disease identification
Early-stage detection before visible symptoms appear
Consistent and scalable monitoring across large agricultural areas
Democratized access to expert-level plant disease diagnostics
Real-time alerts and preventive measure recommendations
By leveraging these technological advances in the field of Artificial Intelligence and Computer vision , we can transform plant disease detection from a reactive process to a proactive system, safeguarding both agricultural productivity and global food security. 
This shift represents not just an agricultural innovation, but a crucial step toward sustainable food production in an increasingly challenging climate.
Applications 

1. Rapid Disease Detection & Early Intervention
Instant disease identification
Early warning system for disease outbreaks
Preventive measures before disease spread
Significant reduction in crop losses through timely action
Cost-effective disease management strategies
2. AI-Powered Knowledge Distribution
Democratizing agricultural expertise through technology
Clear, simplified explanations of complex plant diseases
Step-by-step treatment recommendations
3. Breaking Language Barriers in Agriculture
Support for multiple regional languages
Region-specific treatment recommendations
Community-driven knowledge sharing
Impact Metrics:
Reduced response time to disease outbreaks
Improved crop yield protection
Enhanced farmer knowledge base
Wider reach across diverse farming communities
Motivation For Selection Of The Project

The plant disease detection project aims to support farming communities by enhancing global food security, improving farmer livelihoods, and showcasing the transformative impact of AI in agriculture. Focused on regions with limited access to agricultural experts, the project leverages AI for timely and accurate disease diagnosis. By offering an accessible and cost-effective solution, it bridges the expertise gap, empowers farmers with actionable insights, and has the potential to save billions in crop losses annually. This initiative reflects a commitment to innovation that uplifts underserved communities. 
Dataset Description And Dataset Source
Overview
The Plant Village dataset is a significant resource designed to aid in the identification and classification of plant diseases. It serves as an essential tool for developing machine learning models to assist farmers and agricultural specialists in detecting diseases in crops, which can help mitigate the risk of crop loss and improve agricultural productivity.
Purpose and Objectives
Objective: The primary goal of the dataset is to provide a rich, high-quality source of plant leaf images to support research and the development of AI models for automatic plant disease detection.
Impact: The dataset is part of a larger effort to use technology to combat food insecurity and improve agricultural outcomes, especially in regions where access to expert plant pathologists is limited.
Data Composition
Total Number of Images: The dataset consists of 54,303 images, making it one of the most extensive datasets available for plant disease research.
Number of Classes: There are 38 unique classes, each representing a combination of plant species and disease conditions. The classes include both healthy and diseased states, ensuring that models can learn to distinguish between the two effectively.
Class Examples
Apple:
Diseases: Apple scab, Black rot, Cedar apple rust
Healthy State: Apple healthy
Corn (Maize):
Diseases: Gray leaf spot, Common rust, Northern leaf blight
Healthy State: Corn healthy
Tomato:
Diseases: Bacterial spot, Early blight, Late blight, Leaf mold, Septoria leaf spot, Spider mites, Target spot, Mosaic virus, Yellow leaf curl virus
Healthy State: Tomato healthy
Potato:
Diseases: Early blight, Late blight
Healthy State: Potato healthy
Grape:
Diseases: Black rot, Black measles, Leaf blight
Healthy State: Grape healthy
Image Characteristics
Image Format: All images are in RGB format, ensuring they capture the full color spectrum, which is crucial for identifying visual symptoms of diseases like discoloration, spots, and leaf texture changes.
Image Resolution: The images have varying resolutions but are typically high-quality, taken in controlled environments to reduce noise and ensure consistent lighting.
Background: Most images have a uniform background to isolate the leaf and make the symptoms more visible, which aids in both human and machine learning analysis.
Data Collection Methodology
Controlled Environment: Images were captured under controlled lighting and background conditions, ensuring that the symptoms of diseases are clearly visible and easily distinguishable.
Variety of Symptoms: The dataset includes images displaying a range of disease symptoms, such as:
Spots and Lesions: Indicative of infections by bacteria, fungi, or viruses.
Discoloration: Such as yellowing or browning of leaves.
Leaf Texture Changes: Curling or wilting due to stress or disease.
Augmentation and Preprocessing
Standardization: While the images in the dataset are preprocessed to some extent, they can still benefit from additional augmentations during training.
Potential Augmentation Strategies: We can apply further augmentations such as:
Rotation: To help the model learn invariant features.
Flipping: To increase variability in leaf orientation.
Color Jittering: To simulate different lighting conditions.
Applications
The Plant Village dataset is highly versatile and has been used in various research areas and practical applications, including:
Machine Learning Models: For training convolutional neural networks (CNNs) and other models to detect and classify plant diseases.
Mobile Applications: Used to develop mobile apps that assist farmers by diagnosing plant diseases using smartphone cameras.
Agricultural Advisory: AI-powered systems that can provide farmers with recommendations on disease management.
Challenges and Considerations
Class Imbalance: Some classes may have significantly more images than others, which could lead to biased model predictions. Techniques like oversampling or class weight adjustments may be necessary to address this issue.
Generalisation: Since the images were taken under controlled conditions, models trained on this dataset may not generalise well to real-world conditions, such as varying lighting, occlusions, or different backgrounds. It is important to test models on field-collected data for a robust evaluation.
Disease Similarities: Some plant diseases have visually similar symptoms, making it challenging for both humans and models to distinguish between them. This necessitates careful feature extraction and model tuning.
How to Access and Use
The dataset can be easily accessed using TensorFlow Datasets:

 import tensorflow_datasets as tfds
 dataset, info = tfds.load("plant_village", split="train", with_info=True, as_supervised=True)
split="train": Specifies that we are loading the training split of the dataset.
as_supervised=True: Loads the dataset in a format where images and labels are returned as (image, label) pairs.
with_info=True: Loads additional information about the dataset, such as class names and number of images.
Current Benchmarks
Benchmark
The PlantVillage dataset serves as a benchmark for evaluating image classification models in plant disease detection. As of 2022, the leading model on this dataset is adaptive minimal ensemble, achieving an accuracy of 100%. Other notable models include µ2Net+ (ViT-L/16) with 99.89% accuracy and DenseNet with 99.88% accuracy. These models demonstrate the effectiveness of deep learning architectures in accurately identifying plant 

Ref: https://paperswithcode.com/sota/image-classification-on-plantvillage 

Proposed Plan
The primary objective of the application is to precisely identify plant diseases and provide a concise textual explanation of the diagnosis. Furthermore, users are encouraged to engage by asking additional questions in their native languages to explore potential treatments and preventive measures for the identified disease. To achieve this, we propose the following approach:

Approach


Plant Disease Detection: Use MobileNetV2, a lightweight convolutional neural network optimized for mobile and embedded vision applications. Pre-trained on the PlantVillage dataset.
Generative Explanations: Integrate COTS GPT-3.5/4 for generating detailed disease explanations and treatment recommendations.
Multilingual Support: Integrate Google translate for translation into regional languages like Hindi, Tamil, and Telugu.
User Interface : Using Gradio, build an intuitive interface to interact with the trained models, and view/hear the generative explanations.

Packages and Tools

Data Processing: TensorFlow, PyTorch,pandas, NumPy
Model Training and Validation: MobileNetV2, GPT 3.5, sklearn
Interface Development: Gradio 
Multilingual Support : Google translate
Deployment: FastAPI for backend services, Docker for containerization
Code Versioning: GitHub
MLOps Tools: MLflow, ArgoCD, Prometheus, Grafana
PaaS: AWS with EKS orchestration
Algorithms
Convolutional Neural Network (CNN) for image classification (MobileNetV2).
Transformer-based model for text generation (GPT-3.5/4).

Metrics
Disease Detection: Accuracy, F1-score, Precision, Recall
Explanations: BLEU and METEOR score for text quality, Human Evaluation for comprehensiveness and accuracy

Stages And Deliverables
Data Preparation
Preprocess plant images (resizing, noise removal, augmentations).
Curate and preprocess agricultural text data for explanations.
Model Implementation
Fine-tune MobileNetV2 for plant disease classification.
Use commercially off the shelf GPT-3.5/4 for generating explanations and remedies.
Interface Development
Build a user-friendly Gradio application for image uploads and diagnosis.
Incorporate query handling for further explanations.
Multilingual Enhancements
Integrate and test multilingual support
Deployment and Operations
Containerize application using Docker
Provision AWS infrastructure to perform model training 
Capture metrics using MLFlow and maintain model versions
Deploy trained models on AWS along with the containerized application
Monitoring and Alerting
Track model metrics using Prometheus and Grafana
Create alerts for highlighting deteriorating performance

Deployment Plan
Key Components
Component
Technology
Description
Frontend (UI)
Gradio
User interface for uploading images and selecting languages.
Backend API
FastAPI
Handles image processing, diagnosis generation, and translations.
Translation API
Google Translate
Converts diagnostic text to selected language.
Containerization
Docker
Standardizes deployment across environments.
Cloud Platform
AWS EC2
Hosts backend and frontend, managed with Docker.
CI/CD Pipeline
GitHub Actions
Automated testing, image building, and deployment.
Monitoring
AWS CloudWatch
Monitors application performance and health.

Backend Development (FastAPI)
FastAPI Endpoint Setup:
Disease Detection: Process plant images and predict diseases using a pre-trained model.
Diagnosis Generation: Use the ChatGPT API to generate diagnostic text based on the detected disease.
Translation: Convert diagnosis text to the selected language using Google Translate or Azure Translator API.
Dockerize FastAPI:Create a Dockerfile to containerize FastAPI, install dependencies, and configure the application.
Build and test the Docker image locally.
Frontend Development
Gradio:
Allow users to upload an image and choose a diagnosis language.
Use HTTP requests to connect with Gradio endpoints for detection, diagnosis, and translation.
Deployment on AWS
Backend (FastAPI) Deployment:
Deploy the FastAPI Docker container on an AWS EC2 instance.
Expose the FastAPI service on port 8000 and ensure security groups allow HTTP/HTTPS traffic.
Optional: Use an S3 bucket for storing images if needed for persistent storage.
Frontend (Gradio) Deployment:
Deploy Streamlit on a separate EC2 instance and connect it to the FastAPI backend.
Expose Gradio’s UI on a public IP or domain.
CI/CD Pipeline (GitHub Actions)
Automated Testing:
Use GitHub Actions to set up test workflows to run unit tests for both FastAPI and Gradio code.
Docker Image Build and Push:
Create workflows to build and push the Docker images to a container registry (e.g., Amazon ECR or Docker Hub).
Deployment Workflow:
Create a GitHub Actions workflow to deploy to AWS using SSH.
Trigger this workflow on successful image build or push.
Configure Multilingual Support
Translation API Integration:
Use the googletrans library or Azure Translator for language support in FastAPI’s /translate_diagnosis endpoint.
Allow translations to languages such as Hindi, Tamil, Bengali, etc., and configure this API to return the translated text back to Gradio.
Monitoring and Maintenance
CloudWatch for Monitoring: Use AWS CloudWatch to track CPU usage, memory, and request latency.
Logging: Log application errors and user activities in Gradio and FastAPI for debugging and user experience analysis.
Security: Use environment variables for sensitive API keys (ChatGPT and translation APIs) and configure API authentication for FastAPI.
Application Architecture:









Network Diagram:




Preliminary Exploratory Data Analysis

The PlantVillage dataset consists of 54303 healthy and unhealthy leaf images divided into 38 categories by species and disease.
Please find below the label and the distribution count of images present in the dataset.

Label
Count
Apple___Apple_scab
630
Apple___Black_rot
621
Apple___Cedar_apple_rust
275
Apple___healthy
1645
Blueberry___healthy
1502
Cherry___healthy
854
Cherry___Powdery_mildew
1052
Corn___Cercospora_leaf_spot Gray_leaf_spot
513
Corn___Common_rust
1192
Corn___healthy
1162
Corn___Northern_Leaf_Blight
985
Grape___Black_rot
1180
Grape___Esca_(Black_Measles)
1383
Grape___healthy
423
Grape___Leaf_blight_(Isariopsis_Leaf_Spot)
1076
Orange___Haunglongbing_(Citrus_greening)
5507
Peach___Bacterial_spot
2297
Peach___healthy
360
Pepper,_bell___Bacterial_spot
997
Pepper,_bell___healthy
1477
Potato___Early_blight
1000
Potato___healthy
152
Potato___Late_blight
1000
Raspberry___healthy
371
Soybean___healthy
5090
Squash___Powdery_mildew
1835
Strawberry___healthy
456
Strawberry___Leaf_scorch
1109
Tomato___Bacterial_spot
2127
Tomato___Early_blight
1000
Tomato___healthy
1591
Tomato___Late_blight
1908
Tomato___Leaf_Mold
952
Tomato___Septoria_leaf_spot
1771
Tomato___Spider_mites Two-spotted_spider_mite
1676
Tomato___Target_Spot
1404
Tomato___Tomato_mosaic_virus
373
Tomato___Tomato_Yellow_Leaf_Curl_Virus
5357
Total Count
54303



Class Distribution of Plant Village Dataset


Sample Images
Expected Outcomes
The project’s efficient AI models and scalable deployment deliver accurate, real-time disease detection and treatment advice. This technology empowers farmers to make informed decisions quickly, increasing crop yield and reducing losses, while multilingual support enhances accessibility, driving overall agricultural productivity and food security. Below are the business impacts that can be achieved.

1. Accurate Disease Detection: The system will deliver high-precision results for identifying plant diseases, even under varying image quality and environmental conditions.
Farmers can trust the diagnosis to make timely and informed decisions.

2. Clear and Actionable Treatment Advice: Generative AI will provide easy-to-understand, practical treatment recommendations tailored to the identified disease.
The advice will focus on actionable steps, enabling farmers to implement solutions effectively.
3. Increased Crop Yield and Reduced Loss: Early and accurate disease detection will minimize crop damage and reduce financial losses.
This will contribute to improved agricultural productivity and food security.

4. User-Friendly and Accessible Interface: The intuitive interface will accommodate farmers with different levels of digital literacy. The system’s design will ensure that even first-time users can easily navigate and benefit from the tool.
Project Demonstration Strategy
Phase 1: Introduction and Overview of the Project
Objective: The project’s goals, methodology, and relevance will be introduced.
Plan:
A brief introduction to the problem of plant disease detection and the challenges farmers face.
The objectives of the project will be outlined, highlighting the use of MobileNetV2 for disease detection and GPT-3.5/4 for generating disease explanations and treatment recommendations.
The key methodology behind integrating these AI models will be explained, including how the system is designed to process images, diagnose diseases, and generate actionable advice.

Phase 2: Walkthrough of the System and Interface
Objective: The user interface and image upload process will be demonstrated.
Plan:
Live Demonstration: The audience will be guided through the process of uploading an image of a plant leaf or crop.
The system’s processing steps will be demonstrated, showing how it applies the MobileNetV2 model for disease detection and generates a disease diagnosis.
The system's ability to handle images of varying quality (e.g., clear vs. low-resolution images) will be demonstrated.
System Functionality Check: Prior to the live demonstration, all aspects of the system (image upload, disease detection, and recommendation generation) will be thoroughly tested to ensure everything is working as expected.

Phase 3: Disease Detection and Explanation Generation
Objective: The disease detection and explanation generation processes will be shown.
Plan:
Disease Detection: The output from the MobileNetV2 model will be presented, showcasing the identified disease and its confidence score.
Generative AI Explanation: GPT-3.5/4 will be shown generating detailed explanations for the detected disease, including symptoms, causes, and recommended treatments or preventive measures.
Examples with different plant diseases (e.g., leaf spot, powdery mildew) will be used to demonstrate how the system handles a variety of plant diseases.

Phase 4: Interactive Query Handling
Objective: The system’s ability to respond to follow-up queries will be demonstrated.
Plan:
The audience will be shown how the system responds to queries such as:
“What are the symptoms of this disease?”
“How can I prevent this disease from affecting other crops?”
The clarity and relevance of the responses generated by the AI will be emphasized, showcasing the system’s potential for providing actionable advice.

Phase 5: Results and Performance Evaluation
Objective: The system’s performance will be evaluated and its effectiveness in disease detection demonstrated.
Plan:
Performance metrics, including accuracy, processing time, and relevance/clarity of generated recommendations, will be presented.
Results will be compared to baseline models or other techniques used for plant disease detection.
Testing results based on real-world plant disease images will be shown to demonstrate the system’s robustness.

Phase 6: Learnings
Objective: The challenges encountered during the development of the system and how they were addressed will be discussed.
Plan:
The key challenges, such as handling varying image quality, ensuring high accuracy in disease detection, and fine-tuning the Generative AI for agricultural language, will be explained.
The solutions implemented, including image preprocessing techniques and model adjustments, will be detailed.
Lessons learned from the development process will be shared, focusing on areas like dataset quality and model evaluation.

Project Timeline



November 2024:
November 18, 2024: Project Plan Submission
November 19–26, 2024: Data Preprocessing
December 2024:
November 27–December 8, 2024 : Model Fine-Tuning
December 9–15, 2024: Model Validation and Hyperparameter Tuning
December 9–28, 2024: Chatbot Integration (LLM & Translator API)
December 9–28, 2024: UI Design & Development: 
December 9–28, 2024: Setup ML Ops
December 29, 2024–January 5, 2025: Final Testing and Dry Run
January 2025:
January 6–10, 2025: Final Demo Preparation: 
January 11, 2025: Final Demo
Leaf Guard Team

2303009
Shailesh Kumar
2304878
Deepak Raina
2303646
Shuvayan Das
2304352
Anand
2303112
Sanil Madathil
2303869
Narasimha Murthy Pujari
2304362
Vasudeva Renjala
2304069
Dilip Rajagopal
2303702
Shubhashree
2303859
Suyash Srivastava
2303303
Aakansha Ajay
2303675
Piyush Sharma


Leaf Guard Coordinator  : Shuvayan Das





