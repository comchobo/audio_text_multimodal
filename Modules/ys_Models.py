from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
import torch.nn as nn
import torch.nn.functional as F

def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class FeatureProjector(nn.Module):
    def __init__(self, input_dim, output_dim, layer_norm_eps=1e-05, dropout=0.0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim, eps=layer_norm_eps)
        self.projection = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states):
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class ys_MultimodalModel(nn.Module):
    def __init__(self, text_model_path = 'embargo',
                 audio_model_path = 'kresnik/wav2vec2-large-xlsr-korean',
                 num_labels=3, target_audio_length=16, normalize='traditional', do_normalize=True, class_weight=True):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer, Wav2Vec2FeatureExtractor
        from deprecated.CustomFeatureExtractor import CustomWav2VecFeatureExtractor
        self.text_model = AutoModel.from_pretrained(text_model_path)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_path)

        self.audio_model = AutoModel.from_pretrained(audio_model_path)
        self.audio_model.feature_projection = nn.Sequential()
        self.audio_model.encoder = nn.Sequential()
        self.audio_projector = FeatureProjector(512, 384)
        self.target_audio_length = target_audio_length

        self.classifier = nn.Linear(384, num_labels)

        self.device=torch.device('cuda')
        self.class_weight = class_weight
        if class_weight is True:
            self.class_weight_tensor = torch.tensor([2, 1.3, 0.5]).to(self.device)
        self.num_labels = num_labels

        if normalize=='traditional':
            self.wave_processor = Wav2Vec2FeatureExtractor.from_pretrained(audio_model_path, do_normalize=do_normalize)
        elif normalize=='custom':
            self.wave_processor = CustomWav2VecFeatureExtractor.from_pretrained(audio_model_path, do_normalize=do_normalize)
        else:
            raise NotImplementedError

    def forward(self,
        text_input_ids = None, text_attention_mask = None, text_token_type_ids = None,
        audio_input_values = None, audio_attention_mask = None,
        output_hidden_states = None, return_dict= None, labels = None):

        batch_size = text_input_ids.size(0)
        text_token_embeddings = self.text_model.embeddings(text_input_ids)
        text_token_embeddings = text_token_embeddings.view(*text_input_ids.shape, -1)
        audio_token_embeddings = self.audio_model.feature_extractor(audio_input_values)
        audio_attention_mask = self.audio_model._get_feature_vector_attention_mask(audio_token_embeddings.shape[1],
                                                                                   audio_attention_mask,
                                                                                   add_adapter=False)
        pooled_audio_token_embeddings = []
        for i in range(batch_size):
            pooled_embeddings = F.adaptive_avg_pool1d(audio_token_embeddings[i][:, :audio_attention_mask[i].sum()],
                                                      self.target_audio_length)  # (512, 16)
            pooled_audio_token_embeddings.append(pooled_embeddings.T)
        pooled_audio_token_embeddings = torch.stack(pooled_audio_token_embeddings, dim=0)
        pooled_audio_token_embeddings = self.audio_projector(pooled_audio_token_embeddings)

        inputs_embeds = torch.cat([pooled_audio_token_embeddings, text_token_embeddings], dim=1)
        pooled_audio_attention_mask = torch.ones(batch_size, self.target_audio_length).long().to(self.device)
        attention_mask = torch.cat([pooled_audio_attention_mask, text_attention_mask], dim=1)

        fused_token_embeddings = self.text_model(inputs_embeds=inputs_embeds,
                                                 attention_mask=attention_mask).last_hidden_state
        embeddings = mean_pooling(fused_token_embeddings, attention_mask)
        logits = self.classifier(embeddings)

        loss = None
        if labels is not None:
            if self.class_weight is True:
                loss_fct = CrossEntropyLoss(self.class_weight_tensor)
            else:
                loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

    def preprocess_text(self, text_list):
        return self.text_tokenizer(text_list, max_length=128, padding=True, truncation=True)

    def get_text_processor(self):
        return self.text_tokenizer

    def preprocess_wave(self, wave_list):
        return self.wave_processor(wave_list)

    def get_wave_processor(self):
        return self.wave_processor