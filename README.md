# Chicken Disease Classification using CNN


Chicken diseases pose a serious challenge to the poultry industry, as they can spread rapidly and significantly impact animal health and productivity. Traditional disease identification methods often rely on manual inspection, which can be time-consuming and prone to error.

This project addresses the problem by using image-based disease detection, where visual patterns present in chicken fecal samples are analyzed using deep learning. By automating the detection process, the system helps in identifying potential diseases at an early stage, enabling faster decision-making and improved disease management.


---

## Project Overview

The Chicken Disease Classification Project is an end-to-end deep learning application designed to identify diseases in chickens through image-based analysis. The primary objective of this project is to assist in early disease detection by analyzing chicken fecal images using a Convolutional Neural Network (CNN).

The project follows a structured and modular machine learning workflow, including data ingestion, preprocessing, model building, training, and evaluation. Configuration and parameter files are used to manage paths and hyperparameters, ensuring reproducibility and scalability. The training pipeline is orchestrated using DVC, allowing efficient experiment tracking and version control.

In addition to model training, the project provides a Flask-based web application that enables users to upload images and receive real-time predictions. This makes the solution suitable for practical deployment as well as academic and research purposes.

Overall, this project demonstrates the application of deep learning and MLOps practices to solve a real-world problem in the poultry industry, focusing on automation, reliability, and ease of use.

---

## Tech Stack

- Python  
- TensorFlow / Keras  
- Convolutional Neural Network (CNN)  
- Flask  
- DVC  
- NumPy, Pandas  
- Matplotlib  

---

## Project Structure

<pre>
Chicken-Disease-Classification-Project
│
├── app.py
├── main.py
├── requirements.txt
├── setup.py
├── params.yaml
├── dvc.yaml
├── scores.json
├── inputImage.jpg
│
├── config/
│   └── config.yaml
│
├── research/
│   ├── 01_data_ingestion.ipynb
│   ├── 02_prepare_base_model.ipynb
│   ├── 03_prepare_callbacks.ipynb
│   ├── 04_training.ipynb
│   ├── 05_model_evaluation.ipynb
│   └── trials.ipynb
│
├── src/
│   └── cnnClassifier/
│       ├── components/
│       ├── config/
│       ├── constants/
│       ├── entity/
│       ├── pipeline/
│       └── utils/
│
├── templates/
│   └── index.html
│
├── .github/
│   └── workflows/
│
├── .dvc/
├── .vscode/
├── LICENSE
└── .gitignore
</pre>

---

## DVC Pipeline Stages

1. Data Ingestion  
2. Prepare Base Model  
3. Model Training  
4. Model Evaluation  

Run the full pipeline:
<pre>dvc repro</pre>

## Clone the Repository

```text
git clone https://github.com/Arnab-Ghosh7/Chicken-Disease-Classification-Project.git
cd Chicken-Disease-Classification-Project
```
## Create Environment & Install Dependencies

```text
conda create -n chicken python=3.8 -y
conda activate chicken
pip install -r requirements.txt
```

### Train the Model
```
python main.py
```

### Run Flask App
```
python app.py
```
### Open in browser:
```
http://127.0.0.1:5000/
```

---

## Model Output

The model predicts whether the chicken fecal image belongs to:
- Healthy
- Diseased

Evaluation metrics are stored in `scores.json`.

---

## Configuration

- `config/config.yaml` contains all path and artifact configurations  
- `params.yaml` contains model hyperparameters  

---

## Author

Arnab Ghosh  

GitHub: https://github.com/Arnab-Ghosh7

---

## License

This project is licensed under the MIT License.


