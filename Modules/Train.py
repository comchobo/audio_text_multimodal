from transformers import Trainer
import torch
import numpy as np


class Wave_Unimodal_trainer:
    def __init__(self, debug=True, freeze=True, do_normalize=True, normalize='traditional', WAVE_MAX_LENGTH=16000*6):
        from .Models import SpeechModel
        self.speechmodel = SpeechModel(freeze=freeze, do_normalize=do_normalize, normalize=normalize,
                                       WAVE_MAX_LENGTH=WAVE_MAX_LENGTH)
        self.debug = debug

    def preprocess_dataset(self, wav_folder_path, labeled_json_file_path, label_map, save_target_path):
        import os
        if os.path.exists(save_target_path):
            print('preprocess already done!')
            return
        else:
            os.makedirs(save_target_path)

        from .Prepare import PrepareDataset
        prepare_dataset = PrepareDataset(label_map)
        prepare_dataset.make_labeled_WavUnimodal_dataset_stepBystep(save_target_path = save_target_path,
                                                                    wav_folder_path=wav_folder_path,
                                                                    labeled_json_file_path=labeled_json_file_path,
                                                                    preprocessor=self.speechmodel.get_processor())

    def load_preprocessed_dataset(self, data_path, split_mode=1):
        import os
        from datasets import load_from_disk, concatenate_datasets
        from tqdm import tqdm
        import random

        if split_mode==1:
            temp_loaded_datasets = []
            temp_loaded_datasets_test = []
            idxes = [x for x in range(len(os.listdir(data_path)))]
            folder_paths = [f'{data_path}/{idx}' for idx in idxes]
            if self.debug is True: folder_paths = folder_paths[:8]
            print('loading preprocessed dataset...')
            for folder_path in tqdm(folder_paths[:-6]):
                temp_loaded_datasets.append(load_from_disk(folder_path))
            self.train_dataset = concatenate_datasets(temp_loaded_datasets)

            for folder_path in tqdm(folder_paths[-6:]):
                temp_loaded_datasets_test.append(load_from_disk(folder_path))
            self.test_dataset = concatenate_datasets(temp_loaded_datasets_test)

        else:
            self.dataset = load_from_disk(data_path)

    def set_arguments(self, training_arguments_file_path=None):
        from transformers import TrainingArguments, HfArgumentParser
        parser = HfArgumentParser(TrainingArguments)
        self.training_args, = parser.parse_json_file(json_file=training_arguments_file_path)

    def train(self):
        print('\nTraining start!')
        from utils.utils import wave_collate_fn
        from evaluate import load
        f1_metric = load("f1")

        def compute_metrics(eval_pred):
            predictions = np.argmax(eval_pred.predictions, axis=1)
            return f1_metric.compute(predictions=predictions, references=eval_pred.label_ids, average="micro")

        trainer = Trainer(
            model=self.speechmodel.model,
            data_collator=wave_collate_fn,
            args=self.training_args,
            compute_metrics=compute_metrics,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            tokenizer=self.speechmodel.get_processor(),
        )
        trainer.train()
        if self.debug is False:
            torch.save(trainer.model.state_dict(), self.training_args.output_dir + '_saved')
            return trainer.state.best_metric
        else:
            return [trainer.state.best_metric, sum([np.prod(p.size()) for p in self.speechmodel.model.parameters()])]


class Text_Unimodal_trainer:
    def __init__(self, debug=False, model_path=''):
        from transformers import ElectraForSequenceClassification, ElectraTokenizerFast
        self.model = ElectraForSequenceClassification.from_pretrained(model_path, num_labels=3)
        self.tokenizer = ElectraTokenizerFast.from_pretrained(model_path)
        self.debug = debug

    def load_and_preprocess_dataset(self, labeled_json_file_path, label_map, save_target_path=None, do_save=False):
        from .Prepare import PrepareDataset
        prepare_dataset = PrepareDataset(label_map)
        self.train_dataset, self.test_dataset = prepare_dataset.make_labeled_TextUnimodal_dataset(
            labeled_json_file_path=labeled_json_file_path, preprocessor=self.tokenizer, debug=self.debug)
        if do_save is True:
            self.train_dataset.save_to_disk('./data/saved_text_train')
            self.test_dataset.save_to_disk('./data/saved_text_test')

    def set_arguments(self, training_arguments_file_path=None):
        from transformers import TrainingArguments, HfArgumentParser
        parser = HfArgumentParser(TrainingArguments)
        self.training_args, = parser.parse_json_file(json_file=training_arguments_file_path)


    def train(self):
        print('\nTraining start!')
        from evaluate import load
        f1_metric = load("f1")

        def compute_metrics(eval_pred):
            predictions = np.argmax(eval_pred.predictions, axis=1)
            return f1_metric.compute(predictions=predictions, references=eval_pred.label_ids, average="micro")

        from transformers import DataCollatorWithPadding
        collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=self.model,
            data_collator=collator,
            args=self.training_args,
            compute_metrics=compute_metrics,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            tokenizer=self.tokenizer,
        )
        trainer.train()
        if self.debug is False:
            torch.save(trainer.model.state_dict(), self.training_args.output_dir + '_saved')
            return trainer.state.best_metric
        else:
            return [trainer.state.best_metric, sum([np.prod(p.size()) for p in self.model.parameters()])]


class Wave_Text_Simple_Multimodal_trainer:
    def __init__(self, text_model_path="kykim/electra-kor-base", wave_model_path="kresnik/wav2vec2-large-xlsr-korean",
                 debug=True, freeze_text_model=False, freeze_wave_model=True, do_normalize=False,
                 normalize='traditional', WAVE_MAX_LENGTH=16000*10):
        from .Models import WaveTextSimpleMultimodalModel
        self.model = WaveTextSimpleMultimodalModel(text_model_path=text_model_path,
                                                   wave_model_path=wave_model_path,
                                                   freeze_text_model=freeze_text_model,
                                                   freeze_wave_model=freeze_wave_model,
                                                   do_normalize=do_normalize,
                                                   normalize=normalize, WAVE_MAX_LENGTH=WAVE_MAX_LENGTH)
        self.debug = debug

    def preprocess_multimodal_dataset(self, wav_folder_path, labeled_json_file_path, label_map, save_target_path):
        import os
        if os.path.exists(save_target_path):
            print('preprocess already done!')
            return
        else:
            os.makedirs(save_target_path)

        from .Prepare import PrepareMultimodalDataset
        prepare_dataset = PrepareMultimodalDataset(label_map)
        prepare_dataset.make_labeled_Multimodal_dataset_stepBystep(save_target_path=save_target_path,
                                                                   wav_folder_path=wav_folder_path,
                                                                   labeled_json_file_path=labeled_json_file_path,
                                                                   text_preprocessor=self.model.get_text_processor(),
                                                                   wave_preprocessor=self.model.get_wave_processor())

    def load_preprocessed_dataset(self, data_path, split_mode=1):
        import os
        from datasets import load_from_disk, concatenate_datasets
        from tqdm import tqdm
        import random

        if split_mode==1:
            temp_loaded_datasets = []
            temp_loaded_datasets_test = []
            idxes = [x for x in range(len(os.listdir(data_path)))]
            folder_paths = [f'{data_path}/{idx}' for idx in idxes]
            random.shuffle(folder_paths)
            if self.debug is True : folder_paths = folder_paths[:8]
            print('loading preprocessed dataset...')
            for folder_path in tqdm(folder_paths[:-2]):
                temp_loaded_datasets.append(load_from_disk(folder_path))
            self.train_dataset = concatenate_datasets(temp_loaded_datasets)

            for folder_path in tqdm(folder_paths[-2:]):
                temp_loaded_datasets_test.append(load_from_disk(folder_path))
            self.test_dataset = concatenate_datasets(temp_loaded_datasets_test)

    def load_preprocessed_Full_dataset(self, data_path, split_mode=1):
        import os
        from datasets import load_from_disk, concatenate_datasets
        from tqdm import tqdm

        if split_mode==1:
            temp_loaded_datasets = []
            temp_loaded_datasets_test = []
            idxes = [x for x in range(len(os.listdir(data_path)))]
            folder_paths = [f'{data_path}/{idx}' for idx in idxes]
            print('loading preprocessed dataset...')
            for folder_path in tqdm(folder_paths[:22]):
                temp_loaded_datasets.append(load_from_disk(folder_path))
            self.train_dataset = concatenate_datasets(temp_loaded_datasets)

            for folder_path in tqdm(folder_paths[22:]):
                temp_loaded_datasets_test.append(load_from_disk(folder_path))
            self.test_dataset = concatenate_datasets(temp_loaded_datasets_test)

    def set_arguments(self, training_arguments_file_path=None):
        from transformers import TrainingArguments, HfArgumentParser
        parser = HfArgumentParser(TrainingArguments)
        self.training_args, = parser.parse_json_file(json_file=training_arguments_file_path)

    def train(self):
        print('\nTraining start!')
        from utils.utils import multimodal_collate_fn
        from evaluate import load
        f1_metric = load("f1")

        def compute_metrics(eval_pred):
            predictions = np.argmax(eval_pred.predictions, axis=1)
            return f1_metric.compute(predictions=predictions, references=eval_pred.label_ids, average="micro")

        trainer = Trainer(
            model=self.model,
            data_collator=multimodal_collate_fn,
            args=self.training_args,
            compute_metrics=compute_metrics,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
        )
        trainer.train()
        if self.debug is False:
            torch.save(trainer.model.state_dict(), self.training_args.output_dir + '_saved')
            return trainer.state.best_metric
        else:
            return [trainer.state.best_metric, sum([np.prod(p.size()) for p in self.model.parameters()])]


class Wave_Text_CrossAttention_Multimodal_trainer:
    def __init__(self, text_model_path, wave_model_path, freeze_text_model=False, freeze_wave_model=True,
                 debug=True, do_normalize=False, normalize='traditional', WAVE_MAX_LENGTH=16000*10):
        from .Models import WaveTextCrossAttentionMultimodalModel
        self.model = WaveTextCrossAttentionMultimodalModel(text_model_path=text_model_path,
                                                           wave_model_path=wave_model_path,
                                                           freeze_text_model=freeze_text_model,
                                                           freeze_wave_model=freeze_wave_model,
                                                           do_normalize=do_normalize,
                                                           normalize=normalize,
                                                           WAVE_MAX_LENGTH=WAVE_MAX_LENGTH)
        self.debug = debug

    def preprocess_multimodal_dataset(self, wav_folder_path, labeled_json_file_path, label_map, save_target_path):
        import os
        if os.path.exists(save_target_path):
            print('preprocess already done!')
            return
        else:
            os.makedirs(save_target_path)

        from .Prepare import PrepareMultimodalDataset
        prepare_dataset = PrepareMultimodalDataset(label_map)
        prepare_dataset.make_labeled_Multimodal_dataset_stepBystep(save_target_path=save_target_path,
                                                                   wav_folder_path=wav_folder_path,
                                                                   labeled_json_file_path=labeled_json_file_path,
                                                                   text_preprocessor=self.model.get_text_processor(),
                                                                   wave_preprocessor=self.model.get_wave_processor())

    def load_preprocessed_dataset(self, data_path, split_mode=1):
        import os
        from datasets import load_from_disk, concatenate_datasets
        from tqdm import tqdm
        import random

        if split_mode==1:
            temp_loaded_datasets = []
            temp_loaded_datasets_test = []
            idxes = [x for x in range(len(os.listdir(data_path)))]
            folder_paths = [f'{data_path}/{idx}' for idx in idxes]
            random.shuffle(folder_paths)
            if self.debug is True : folder_paths = folder_paths[:8]
            print('loading preprocessed dataset...')
            for folder_path in tqdm(folder_paths[:-2]):
                temp_loaded_datasets.append(load_from_disk(folder_path))
            self.train_dataset = concatenate_datasets(temp_loaded_datasets)

            for folder_path in tqdm(folder_paths[-2:]):
                temp_loaded_datasets_test.append(load_from_disk(folder_path))
            self.test_dataset = concatenate_datasets(temp_loaded_datasets_test)


    def load_preprocessed_Full_dataset(self, data_path, split_mode=1):
        import os
        from datasets import load_from_disk, concatenate_datasets
        from tqdm import tqdm

        temp_loaded_datasets = []
        temp_loaded_datasets_test = []
        idxes = [x for x in range(len(os.listdir(data_path)))]
        folder_paths = [f'{data_path}/{idx}' for idx in idxes]
        print('loading preprocessed dataset...')
        for folder_path in tqdm(folder_paths[:22]):
            temp_loaded_datasets.append(load_from_disk(folder_path))
        self.train_dataset = concatenate_datasets(temp_loaded_datasets)

        for folder_path in tqdm(folder_paths[22:]):
            temp_loaded_datasets_test.append(load_from_disk(folder_path))
        self.test_dataset = concatenate_datasets(temp_loaded_datasets_test)

    def set_arguments(self, training_arguments_file_path=None):
        from transformers import TrainingArguments, HfArgumentParser
        parser = HfArgumentParser(TrainingArguments)
        self.training_args, = parser.parse_json_file(json_file=training_arguments_file_path)

    def train(self):
        print('\nTraining start!')
        from utils.utils import multimodal_collate_fn
        from evaluate import load
        f1_metric = load("f1")

        def compute_metrics(eval_pred):
            predictions = np.argmax(eval_pred.predictions, axis=1)
            return f1_metric.compute(predictions=predictions, references=eval_pred.label_ids, average="micro")

        trainer = Trainer(
            model=self.model,
            data_collator=multimodal_collate_fn,
            args=self.training_args,
            compute_metrics=compute_metrics,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
        )
        trainer.train()
        if self.debug is False:
            torch.save(trainer.model.state_dict(), self.training_args.output_dir + '_saved')
            return trainer.state.best_metric
        else:
            return [trainer.state.best_metric, sum([np.prod(p.size()) for p in self.model.parameters()])]


class ys_Trainer:
    def __init__(self, text_model_path="microsoft/mdeberta-v3-base", wave_model_path="facebook/wav2vec2-large-xlsr-53"
                 , debug=True, do_normalize=False, freeze_text_model=False, freeze_wave_model=True,
                 WAVE_MAX_LENGTH=16000*10):
        from .ys_Models import ys_MultimodalModel
        self.model = ys_MultimodalModel()
        self.debug = debug

    def preprocess_multimodal_dataset(self, wav_folder_path, labeled_json_file_path, label_map, save_target_path):
        import os
        if os.path.exists(save_target_path):
            print('preprocess already done!')
            return
        else:
            os.makedirs(save_target_path)

        from .Prepare import PrepareMultimodalDataset
        prepare_dataset = PrepareMultimodalDataset(label_map)
        prepare_dataset.make_labeled_Multimodal_dataset_stepBystep(save_target_path=save_target_path,
                                                                   wav_folder_path=wav_folder_path,
                                                                   labeled_json_file_path=labeled_json_file_path,
                                                                   text_preprocessor=self.model.get_text_processor(),
                                                                   wave_preprocessor=self.model.get_wave_processor())

    def load_preprocessed_dataset(self, data_path, split_mode=1):
        import os
        from datasets import load_from_disk, concatenate_datasets
        from tqdm import tqdm
        import random

        if split_mode==1:
            temp_loaded_datasets = []
            temp_loaded_datasets_test = []
            idxes = [x for x in range(len(os.listdir(data_path)))]
            folder_paths = [f'{data_path}/{idx}' for idx in idxes]
            random.shuffle(folder_paths)
            if self.debug is True : folder_paths = folder_paths[:8]
            print('loading preprocessed dataset...')
            for folder_path in tqdm(folder_paths[:-2]):
                temp_loaded_datasets.append(load_from_disk(folder_path))
            self.train_dataset = concatenate_datasets(temp_loaded_datasets)

            for folder_path in tqdm(folder_paths[-2:]):
                temp_loaded_datasets_test.append(load_from_disk(folder_path))
            self.test_dataset = concatenate_datasets(temp_loaded_datasets_test)

    def load_preprocessed_Full_dataset(self, data_path, split_mode=1):
        import os
        from datasets import load_from_disk, concatenate_datasets
        from tqdm import tqdm

        if split_mode==1:
            temp_loaded_datasets = []
            temp_loaded_datasets_test = []
            idxes = [x for x in range(len(os.listdir(data_path)))]
            folder_paths = [f'{data_path}/{idx}' for idx in idxes]
            print('loading preprocessed dataset...')
            for folder_path in tqdm(folder_paths[:22]):
                temp_loaded_datasets.append(load_from_disk(folder_path))
            self.train_dataset = concatenate_datasets(temp_loaded_datasets)

            for folder_path in tqdm(folder_paths[22:]):
                temp_loaded_datasets_test.append(load_from_disk(folder_path))
            self.test_dataset = concatenate_datasets(temp_loaded_datasets_test)

    def set_arguments(self, training_arguments_file_path=None):
        from transformers import TrainingArguments, HfArgumentParser
        parser = HfArgumentParser(TrainingArguments)
        self.training_args, = parser.parse_json_file(json_file=training_arguments_file_path)

    def train(self):
        print('\nTraining start!')
        from utils.utils import multimodal_collate_fn
        from evaluate import load
        f1_metric = load("f1")

        def compute_metrics(eval_pred):
            predictions = np.argmax(eval_pred.predictions, axis=1)
            return f1_metric.compute(predictions=predictions, references=eval_pred.label_ids, average="micro")

        trainer = Trainer(
            model=self.model,
            data_collator=multimodal_collate_fn,
            args=self.training_args,
            compute_metrics=compute_metrics,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
        )
        trainer.train()
        if self.debug is False:
            torch.save(trainer.model.state_dict(), self.training_args.output_dir + '_saved')
            return trainer.state.best_metric
        else:
            return [trainer.state.best_metric, sum([np.prod(p.size()) for p in self.model.parameters()])]


if __name__ == '__main__':
    pass

