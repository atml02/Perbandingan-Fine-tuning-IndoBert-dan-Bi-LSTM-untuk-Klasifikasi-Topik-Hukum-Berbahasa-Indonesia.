# import torch
# import os
# import numpy as np

# class Predictor:
#     UPLOAD_FOLDER = 'uploads'
#     ALLOWED_EXTENSIONS = {'csv'}
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     MAX_SEQ_LEN = 128
    
#     LABEL_MAPPING = {
#         0: 'Pidana',
#         1: 'Ketenagakerjaan',
#         2: 'Keluarga',
#         3: 'Kenegaraan',
#         4: 'Bisnis',
#         5: 'Perdata',
#         6: 'Pertanahan & Properti',
#         7: 'Kekayaan Intelektual',
#         8: 'Ilmu Hukum',
#         9: 'Perlindungan Konsumen',
#         10: 'Profesi Hukum',
#         11: 'Teknologi',
#         12: 'Hak Asasi Manusia',
#     }
    
#     I2W = {index: label for index, label in LABEL_MAPPING.items()}

#     def __init__(self, model_handler):
#         self.model_handler = model_handler
        
#     def predict_sentence(self, model, sentence):
#         """Predict topic for a single sentence."""
#         model.eval()
        
#         inputs = self.model_handler.tokenizer(
#             sentence,
#             return_tensors='pt',
#             padding=True,
#             truncation=True,
#             max_length=self.MAX_SEQ_LEN
#         )
#         inputs = {key: val.to(self.DEVICE) for key, val in inputs.items()}
        
#         with torch.no_grad():
#             outputs = model(**inputs)
#             logits = outputs.logits
#             probabilities = torch.softmax(logits, dim=1)
#             predicted_class = torch.argmax(probabilities, dim=1).item()
        
#         predicted_label = self.I2W[predicted_class]
        
#         return predicted_class, predicted_label, probabilities[0].cpu().numpy()
    
import torch
import os
import numpy as np
import re

class Predictor:
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'csv'}
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_SEQ_LEN = 128
    
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

    # Sanitizer aman untuk HTML ID
    @staticmethod
    def sanitize_label(label):
        label = label.replace("&", "and")   # ubah '&' jadi 'and'
        label = label.replace(" ", "_")     # ubah spasi jadi _
        label = re.sub(r'[^a-zA-Z0-9_]', '', label)  # hapus karakter asing
        return label

    # Mapping index → label asli
    I2W = {index: label for index, label in LABEL_MAPPING.items()}

    # Mapping index → label yang aman untuk HTML
    I2W_SAFE = {index: None for index in LABEL_MAPPING}  # placeholder

    def __init__(self, model_handler):
        self.model_handler = model_handler

        # Isi mapping aman ketika instance dibuat
        for idx, label in self.LABEL_MAPPING.items():
            self.I2W_SAFE[idx] = self.sanitize_label(label)
        
    def predict_sentence(self, model, sentence):
        """Predict topic for a single sentence."""
        model.eval()

        inputs = self.model_handler.tokenizer(
            sentence,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.MAX_SEQ_LEN
        )
        inputs = {key: val.to(self.DEVICE) for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            # logits = outputs.logits
            # Kalau output adalah HuggingFace model → ada atribut logits
            if hasattr(outputs, "logits"):
                logits = outputs.logits

            # Kalau output adalah tuple dari BiLSTM → unpack manual
            elif isinstance(outputs, tuple):
                _, logits = outputs

            else:
                raise ValueError("Unknown model output format!")

            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
        
        return {
            "class": predicted_class,
            "label": self.I2W[predicted_class],
            "label_safe": self.I2W_SAFE[predicted_class],
            "probabilities": probabilities[0].cpu().numpy()
        }
