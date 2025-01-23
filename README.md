# DelayGuard

**DelayGuard** is a Streamlit-based web application designed to predict the likelihood of delays in capital projects. It uses a machine learning model trained on project data to provide probabilistic predictions and qualitative insights.

⭐️
[![GitHub stars](https://img.shields.io/github/stars/okaeiz/delayguard?style=social)](https://github.com/okaeiz/delayguard/stargazers)
⭐️

## Repository Structure

```bash
.
├── Capital_Project_Schedules_and_Budgets.xlsx # Dataset used for training the model
├── README.md  # Project's readme file
├── app.py  # Streamlit application code
├── decision_tree_model_with_smote.pkl # Pre-trained machine learning model
├── project_report.pdf # Project's short report
└── proposal.pdf # Project's initial proposal
```

---

## Features

- **User-Friendly Interface**: Built with Streamlit, the app provides an intuitive and interactive interface.
- **Delay Prediction**: Predicts the probability of project delays based on user inputs.
- **Qualitative Insights**: Provides a qualitative interpretation of the predicted probability (e.g., "خیلی کم احتمال", "محتمل").
- **Dataset Overview**: Displays a sample of the dataset and key statistics in the sidebar.
- **Model Performance**: Shows the model's accuracy, confusion matrix, and classification report.

---

## Prerequisites

Before running the app, ensure you have the following installed:

- Python 3.8 or higher
- Streamlit
- Pandas
- Scikit-learn
- Imbalanced-learn (imblearn)
- Joblib

You can install the required packages using the following command:

```bash
pip install streamlit pandas scikit-learn imbalanced-learn joblib
```

---

## Setup and Usage

### 1. Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/okaeiz/DelayGuard.git
cd DelayGuard
```

### 2. Run the Streamlit App

Run the app using the following command:

```bash
streamlit run app.py
```

This will start the Streamlit server and open the app in your default web browser.

---

## How to Use the App

1. **Input Project Details**:
   - Select the **Project Geographic District** from the dropdown menu.
   - Choose the **Project Type** from the available options.
   - Specify the **Project Phase** and **Project Status**.
   - Enter the **Project Budget Amount**.

2. **Predict Delay Probability**:
   - Click the **پیش‌بینی** (Predict) button to see the results.

3. **View Results**:
   - The app will display the **Probability of Delay** (as a percentage) and the **Qualitative Output** (e.g., "خیلی کم احتمال", "محتمل").

4. **Explore Additional Data**:
   - Use the sidebar to view:
     - A sample of the dataset.
     - Key statistics about the dataset.
     - Model performance metrics (accuracy, confusion matrix, and classification report).

---

## Dataset

The dataset used for training the model is stored in `Capital_Project_Schedules_and_Budgets.xlsx`. It contains the following key columns:

- **Project Geographic District**: The geographic district of the project.
- **Project Type**: The type of project (e.g., educational, infrastructure).
- **Project Phase Name**: The current phase of the project (e.g., design, construction).
- **Project Status Name**: The status of the project (e.g., in-progress, completed).
- **Project Budget Amount**: The budget allocated for the project.
- **Delay**: The target variable indicating whether the project was delayed (1) or not (0).

---

## Model

The machine learning model is a **Random Forest Classifier** trained on the dataset. It uses **SMOTE (Synthetic Minority Over-sampling Technique)** to handle class imbalance. The trained model is saved as `decision_tree_model_with_smote.pkl` and loaded into the app for predictions.

### Model Performance

- **Accuracy**: The model's accuracy on the test set.
- **Confusion Matrix**: A matrix showing true vs. predicted delays.
- **Classification Report**: Precision, recall, and F1-score for each class.

---

## Customization

- **Change Browser Tab Title**: Modify the JavaScript code in `app.py` to update the browser tab title.
- **Add New Features**: Extend the app by adding new input fields or visualizations.
- **Update Dataset**: Replace `Capital_Project_Schedules_and_Budgets.xlsx` with a new dataset and retrain the model.

---

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes.
4. Submit a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or feedback, feel free to reach out:

- **GitHub**: [okaeiz](https://github.com/okaeiz)
- **Email**: [ho3kaei@gmail.com]

---

## Acknowledgments
- Libraries:
  - **Streamlit**: For providing an excellent framework for building data apps.
  - **Scikit-learn**: For the machine learning tools and algorithms.
  - **Imbalanced-learn**: For handling class imbalance in the dataset.
- People:
  - **Mohammad Tahsildoust**: The main lecturer who convinced me how beautiful ML actually is.
  - **Farbod Shahverdi**: The assistant who taught me the technical aspects.

---

Enjoy using **DelayGuard**! 🚀
