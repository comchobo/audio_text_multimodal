import torch
from torch import nn, cat
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.activations import get_activation


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


from deprecated.CustomFeatureExtractor import CustomWav2VecFeatureExtractor
class SpeechModel(nn.Module):
    def __init__(self, model_path="kresnik/wav2vec2-large-xlsr-korean", freeze=True, do_normalize=False,
                 normalize='traditional', WAVE_MAX_LENGTH=16000*10):
        super().__init__()
        from transformers import Wav2Vec2ForSequenceClassification
        from transformers import Wav2Vec2FeatureExtractor
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path, num_labels=3)
        if freeze is True:
            self.model.freeze_feature_encoder()
        if normalize=='traditional':
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_path, do_normalize=do_normalize,
                                                                      max_length=WAVE_MAX_LENGTH)
        elif normalize=='custom':
            self.processor = CustomWav2VecFeatureExtractor.from_pretrained(model_path, do_normalize=do_normalize,
                                                                           max_length=WAVE_MAX_LENGTH)
        else:
            raise NotImplementedError

    def forward(self, input_value=None, attention_mask=None, labels=None, return_dict=False):
        logits = self.model(input_value, attention_mask, labels)

        loss = None
        if labels is not None:
            if self.class_weight is True:
                loss_fct = CrossEntropyLoss(weight=self.class_weight_tensor)
            else:
                loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

    def preprocess(self, wave_list):
        return self.processor(wave_list)

    def get_processor(self):
        return self.processor


class MultimodalSimpleMLPHead(nn.Module):
    def __init__(self, cat_hidden_sizes, text_hidden_sizes, num_labels):
        super().__init__()
        self.text_dense = nn.Linear(text_hidden_sizes, text_hidden_sizes)
        self.out_proj = nn.Linear(cat_hidden_sizes, num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, text_hidden, audio_hidden, **kwargs):
        text_hidden = text_hidden[:, 0, :]
        text_hidden = self.dropout(text_hidden)
        text_hidden = self.text_dense(text_hidden)
        text_hidden = get_activation("gelu")(text_hidden)
        text_hidden = self.dropout(text_hidden)

        x = cat((text_hidden, audio_hidden), 1)
        x = self.out_proj(x)
        return x


class CA_MultimodalMLPHead(nn.Module):
    def __init__(self, text_hidden_sizes, num_labels):
        super().__init__()
        self.dense = nn.Linear(2 * text_hidden_sizes, text_hidden_sizes)
        self.out_proj = nn.Linear(text_hidden_sizes, num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, text_hidden, audio_hidden, **kwargs):
        x = cat((text_hidden, audio_hidden), 1)
        x = self.dropout(x)
        x = self.dense(x)
        x = get_activation("gelu")(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class WaveTextSimpleMultimodalModel(nn.Module):
    def __init__(self, text_model_path="kykim/electra-kor-base", wave_model_path="kresnik/wav2vec2-large-xlsr-korean"
                 , freeze_text_model=False, freeze_wave_model=True, num_labels=3, do_normalize=True, normalize='traditional',
                 class_weight=True, WAVE_MAX_LENGTH=16000*10):
        super().__init__()
        from transformers import ElectraTokenizerFast, ElectraModel, ElectraForSequenceClassification
        self.text_model = ElectraForSequenceClassification.from_pretrained(text_model_path, num_labels=3)
        self.text_model.load_state_dict(torch.load('./finetuned_models/text_unimodal_test_quarter_saved'))

        self.text_tokenizer = ElectraTokenizerFast.from_pretrained(text_model_path) # electra_base output = 768
        text_hid_dim = self.text_model.electra.embeddings.word_embeddings.embedding_dim

        from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
        self.wave_model = Wav2Vec2Model.from_pretrained(wave_model_path)

        if normalize=='traditional':
            self.wave_processor = Wav2Vec2FeatureExtractor.from_pretrained(wave_model_path, do_normalize=do_normalize,
                                                                           max_length=WAVE_MAX_LENGTH)
        elif normalize=='custom':
            self.wave_processor = CustomWav2VecFeatureExtractor.from_pretrained(wave_model_path,
                                                                                do_normalize=do_normalize,
                                                                                max_length=WAVE_MAX_LENGTH)
        else:
            raise NotImplementedError

        wave_mid_dim = self.wave_model.config.hidden_size
        wave_hid_dim = self.wave_model.config.classifier_proj_size
        self.wave_projector = nn.Linear(wave_mid_dim, wave_hid_dim) # wave2vec output = 1024, proj = 256

        self.classifier = MultimodalSimpleMLPHead(text_hidden_sizes=text_hid_dim,
                                                  cat_hidden_sizes=wave_hid_dim+text_hid_dim,
                                                  num_labels=num_labels)
        self.device = torch.device('cuda')
        self.class_weight = class_weight
        if class_weight is True:
            self.class_weight_tensor = torch.tensor([2, 1.3, 0.5]).to(self.device)
        self.num_labels = num_labels

        if freeze_text_model is True:
            for param in self.text_model.parameters():
                param.requires_grad = False
        if freeze_wave_model is True:
            self.wave_model.freeze_feature_encoder()

    def forward(self,
        text_input_ids = None, text_attention_mask = None, text_token_type_ids = None,
        audio_input_values = None, audio_attention_mask = None,
        output_hidden_states = None, return_dict= None, labels = None):

        electra_hidden_states = self.text_model.electra(
            text_input_ids,
            attention_mask=text_attention_mask,
            token_type_ids=text_token_type_ids,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)
        electra_outputs = electra_hidden_states[0]

        wave2vec_outputs = self.wave_model(
            audio_input_values,
            attention_mask=audio_attention_mask,
            output_attentions=None,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        wave2vec_outputs_last = wave2vec_outputs[0]
        wave2vec_outputs_last = self.wave_projector(wave2vec_outputs_last)
        padding_mask = self.wave_model._get_feature_vector_attention_mask(wave2vec_outputs_last.shape[1], audio_attention_mask)
        wave2vec_outputs_last[~padding_mask] = 0.0
        wave2vec_pooled_outputs = wave2vec_outputs_last.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

        logits = self.classifier(electra_outputs, wave2vec_pooled_outputs)

        loss = None
        if labels is not None:
            if self.class_weight is True:
                loss_fct = CrossEntropyLoss(weight=self.class_weight_tensor)
            else:
                loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + wave2vec_outputs[2:]
            return ((loss,) + output) if loss is not None else output

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


class WaveTextCrossAttentionMultimodalModel(nn.Module):
    def __init__(self, text_model_path="kykim/electra-kor-base", wave_model_path="kresnik/wav2vec2-large-xlsr-korean",
                 freeze_text_model=False, freeze_wave_model=True, num_labels=3, do_normalize=True, normalize='traditional',
                 class_weight=True, WAVE_MAX_LENGTH=16000*10):
        super().__init__()
        from transformers import ElectraModel, ElectraTokenizerFast
        self.text_model = ElectraModel.from_pretrained(text_model_path)
        self.text_tokenizer = ElectraTokenizerFast.from_pretrained(text_model_path) # electra_base output = 768
        text_hid_dim = self.text_model.embeddings.word_embeddings.embedding_dim

        from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
        self.wave_model = Wav2Vec2Model.from_pretrained(wave_model_path)

        if normalize == 'traditional':
            self.wave_processor = Wav2Vec2FeatureExtractor.from_pretrained(wave_model_path, do_normalize=do_normalize,
                                                                           max_length=WAVE_MAX_LENGTH)
        elif normalize == 'custom':
            self.wave_processor = CustomWav2VecFeatureExtractor.from_pretrained(wave_model_path,
                                                                                do_normalize=do_normalize,
                                                                                max_length=WAVE_MAX_LENGTH)
        else:
            raise NotImplementedError

        wave_mid_dim = self.wave_model.config.hidden_size
        self.wave_projector = nn.Linear(wave_mid_dim, text_hid_dim) # wave2vec output = 1024, proj = 256

        self.classifier = CA_MultimodalMLPHead(text_hidden_sizes=text_hid_dim,
                                                  num_labels=num_labels)
        self.device = torch.device('cuda')
        self.class_weight=class_weight
        if class_weight is True:
            self.class_weight_tensor = torch.tensor([2, 1.3, 0.5]).to(self.device)
        self.num_labels = num_labels

        from torch.nn import MultiheadAttention
        self.cross_attention = MultiheadAttention(text_hid_dim, 8, batch_first=True) # text_attention_heads

        if freeze_text_model is True:
            for param in self.text_model.parameters():
                param.requires_grad = False
        if freeze_wave_model is True:
            self.wave_model.freeze_feature_encoder()

    def forward(self,
        text_input_ids = None, text_attention_mask = None, text_token_type_ids = None,
        audio_input_values = None, audio_attention_mask = None,
        output_hidden_states = None, return_dict= None, labels = None):

        electra_hidden_states = self.text_model(
            text_input_ids,
            attention_mask=text_attention_mask,
            token_type_ids=text_token_type_ids,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)

        wave2vec_outputs = self.wave_model(
            audio_input_values,
            attention_mask=audio_attention_mask,
            output_attentions=None,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        electra_outputs_last = electra_hidden_states[0]
        wave2vec_outputs_last = wave2vec_outputs[0]
        text_attention_mask_for_outputs = text_attention_mask.le(0.5)
        audio_attention_mask_for_outputs = self.wave_model._get_feature_vector_attention_mask(
            wave2vec_outputs_last.shape[1], audio_attention_mask)
        wave2vec_outputs_last = self.wave_projector(wave2vec_outputs_last)

        attn_output_text, _ = self.cross_attention(electra_outputs_last, wave2vec_outputs_last, wave2vec_outputs_last,
                                                   key_padding_mask=~audio_attention_mask_for_outputs)
        attn_output_audio, _ = self.cross_attention(wave2vec_outputs_last, electra_outputs_last, electra_outputs_last,
                                                   key_padding_mask=text_attention_mask_for_outputs)
        attn_output_text += electra_outputs_last
        attn_output_audio += wave2vec_outputs_last

        text_pooled_outputs = mean_pooling(attn_output_text, text_attention_mask)
        wave2vec_pooled_outputs = mean_pooling(attn_output_audio, audio_attention_mask_for_outputs)
        # text_pooled_outputs = attn_output_text[:, 0, :]
        # wave2vec_pooled_outputs = attn_output_audio[:, 0, :]
        logits = self.classifier(text_pooled_outputs, wave2vec_pooled_outputs)

        loss = None
        if labels is not None:
            if self.class_weight is True:
                loss_fct = CrossEntropyLoss(weight=self.class_weight_tensor)
            else:
                loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + wave2vec_outputs[2:]
            return ((loss,) + output) if loss is not None else output

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
