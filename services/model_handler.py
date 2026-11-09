# import torch
# from transformers import BertTokenizer, BertForSequenceClassification

# class ModelHandler:
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     MODEL_TOKENIZER_MAPPING = {
#         'model1': {
#             'model_checkpoint_path': 'static/model/best_model_1.th'
#         },
#         'model2': {
#             'model_checkpoint_path': 'static/model/best_model_2.th'
#         },
#         'model3': {
#             'model_checkpoint_path': 'static/model/best_model_3.th'
#         },
#         'model4': {
#             'model_checkpoint_path': 'static/model/best_model_4.th'
#         },
#         'model5': {
#             'model_checkpoint_path': 'static/model/best_model_5.th'
#         },
#         'model6': {
#             'model_checkpoint_path': 'static/model/best_model_6.th'
#         },
#         'model7': {
#             'model_checkpoint_path': 'static/model/best_model_7.th'
#         },
#         'model8': {
#             'model_checkpoint_path': 'static/model/best_model_8.th'
#         }
#     }
    
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

#     def __init__(self):
#         self.tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
        
#     def load_model(self, model_selection):
#         """Load the model based on the selected model."""
#         model_info = self.MODEL_TOKENIZER_MAPPING[model_selection]
#         model = BertForSequenceClassification.from_pretrained(
#             'indobenchmark/indobert-base-p1', 
#             num_labels=len(self.LABEL_MAPPING)
#         )
#         model.load_state_dict(
#             torch.load(model_info['model_checkpoint_path'], 
#             map_location=self.DEVICE), 
#             strict=False
#         )
#         model.to(self.DEVICE)
#         model.eval()
#         return model

import torch
from transformers import BertTokenizer, BertForSequenceClassification
from models.bilstm_classifier import BiLSTMClassifier   # pastikan path benar

class ModelHandler:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    MODEL_TOKENIZER_MAPPING = {
        'model1': {'type': 'bert', 'model_checkpoint_path': 'static/model/best_model_1.th'},
        'model2': {'type': 'bert', 'model_checkpoint_path': 'static/model/best_model_2.th'},
        'model3': {'type': 'bert', 'model_checkpoint_path': 'static/model/best_model_3.th'},
        'model4': {'type': 'bert', 'model_checkpoint_path': 'static/model/best_model_4.th'},

        'model5': {'type': 'bilstm', 'model_checkpoint_path': 'static/model/best_model_5.th'},
        'model6': {'type': 'bilstm', 'model_checkpoint_path': 'static/model/best_model_6.th'},
        'model7': {'type': 'bilstm', 'model_checkpoint_path': 'static/model/best_model_7.th'},
        'model8': {'type': 'bilstm', 'model_checkpoint_path': 'static/model/best_model_8.th'},
    }
    
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

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
        
    def load_model(self, model_selection):
        info = self.MODEL_TOKENIZER_MAPPING[model_selection]
        model_type = info['type']
        path = info['model_checkpoint_path']

        # ---------------------------
        # Load IndoBERT model
        # ---------------------------
        if model_type == 'bert':
            model = BertForSequenceClassification.from_pretrained(
                'indobenchmark/indobert-base-p1',
                num_labels=len(self.LABEL_MAPPING)
            )
            model.load_state_dict(
                torch.load(path, map_location=self.DEVICE), strict=False
            )
            model.to(self.DEVICE)
            model.eval()
            return model

        # ---------------------------
        # Load BiLSTM model
        # ---------------------------
        elif model_type == 'bilstm':
            model = BiLSTMClassifier(
                vocab_size=self.tokenizer.vocab_size,  # pakai vocab BERT
                embed_dim=300,
                hidden_size=256,
                num_layers=2,
                num_labels=len(self.LABEL_MAPPING),
                dropout=0.3
            )

            model.load_state_dict(torch.load(path, map_location=self.DEVICE))
            model.to(self.DEVICE)
            model.eval()
            return model
