import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

class Evaluator:
    LABEL_MAPPING = {
        0: 'Pidana',
        1: 'Ketenagakerjaan',
        2: 'Keluarga',
        3: 'Kenegaraan',
        4: 'Bisnis',
        5: 'Perdata',
        6: 'Pertanahan & Properti',
        7: 'Kekayaan Intelektual',
        8: 'Ilmu Hukum',
        9: 'Perlindungan Konsumen',
        10: 'Profesi Hukum',
        11: 'Teknologi',
        12: 'Hak Asasi Manusia',
    }
    
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'csv'}

    def __init__(self, model_handler, predictor):
        self.model_handler = model_handler
        self.predictor = predictor
        
    def load_and_preprocess_dataset(self, file):
        """Load the CSV dataset."""
        try:
            data = pd.read_csv(file)
            if 'question' not in data.columns or 'label' not in data.columns:
                raise ValueError("CSV must contain 'label' and 'question' columns")
            
            return data['label'].tolist(), data['question'].tolist()
        
        except Exception as e:
            print(f"Error processing the dataset: {e}")
            return [], []
    
    def evaluate_model(self, model_selection, file_path):
        """Evaluate the selected model on the uploaded CSV file."""
        # Default to model 1 if no selection is provided
        if model_selection is None:
            print("No model selected. Using default model 1.")
            model_selection = 1
            
        model = self.model_handler.load_model(model_selection)
        true_labels, questions = self.load_and_preprocess_dataset(file_path)

        if not true_labels or not questions:
            return None, None, None, None, None, None
        
        predicted_labels = []
        all_probabilities = []
        
        for question in questions:
            result = self.predictor.predict_sentence(model, question)

            predicted_labels.append(result["label"])
            all_probabilities.append(result["probabilities"])

        
        accuracy = accuracy_score(true_labels, predicted_labels)
        
        unique_labels = np.unique(true_labels)
        report = classification_report(true_labels, predicted_labels, labels=unique_labels, output_dict=True)
        
        precision = report['macro avg']['precision']
        recall = report['macro avg']['recall']
        f1 = report['macro avg']['f1-score']
        
        class_metrics = {
            label: {
                'precision': report[label]['precision'],
                'recall': report[label]['recall'],
                'f1_score': report[label]['f1-score']
            } for label in unique_labels
        }
        
        df = pd.DataFrame({
            'label': true_labels,
            'question': questions,
            'predicted_label': predicted_labels,
            'probabilities': all_probabilities
        })
        
        # Randomize rows without using a fixed seed for truly random results
        return accuracy, precision, recall, f1, class_metrics, df.sample(15, random_state=None)
