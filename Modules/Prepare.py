import soundfile as sf
WAVE_MAX_LENGTH = 16000*6

class PrepareDataset:
    def __init__(self, label_map):
        self.label_map = label_map

    def prepare_wav_unimodal_labeled(self, wav_folder_path, labeled_json_file_path, start_idx=-1, end_idx=-1):
        import os
        if start_idx!=-1 and end_idx!=-1:
            idxes = [str(x) for x in range(int(start_idx), int(end_idx))]
        else:
            idxes = [str(x) for x in range(1, len(os.listdir(wav_folder_path))+1)]

        import json
        wav_files = [f'{wav_folder_path}/clip_{idx}.wav' for idx in idxes]
        json_files = [f'{labeled_json_file_path}/clip_{idx}.json' for idx in idxes]
        wave_list=[]
        label_list=[]

        for wav_file, json_file in zip(wav_files, json_files):
            if os.path.isfile(wav_file) is False: continue

            wave_instance, _ = sf.read(wav_file)
            with open(json_file,'r',encoding='utf-8') as f:
                json_instance = json.load(f)

                for mention in json_instance:
                    start = round(mention['start_wav_time'])
                    end = round(mention['end_wav_time'])
                    label = self.label_map[mention['emotion']['multimodal']['emotion']]
                    if label == -1:
                        continue
                    if len(wave_instance[start:end]) == 0:
                        print('wave split error!')
                        continue
                    wave_list.append(wave_instance[start:end])
                    label_list.append(label)
                # import sounddevice as sd
                # sd.play(wave_instance[start:end], 16000) -> you can try playing with this, to check wave is well split

        return wave_list, label_list

    def make_labeled_WavUnimodal_dataset(self, wav_folder_path, labeled_json_file_path, preprocessor):
        print('\nReading the data...')
        global WAVE_MAX_LENGTH
        wave_list, label_list = self.prepare_wav_unimodal_labeled(wav_folder_path=wav_folder_path
                                                                  , labeled_json_file_path=labeled_json_file_path)
        from datasets import Dataset
        temp_dataset = Dataset.from_dict({'wave':wave_list})

        def preprocess(row):
            return preprocessor(row['wave'], truncation=True, padding="longest", max_length=WAVE_MAX_LENGTH
                                , sampling_rate=16000)

        print('\nTokenizing data...')
        preprocessed_wave_list = temp_dataset.map(preprocess, batched=True, batch_size=100)
        preprocessed_wave_list = preprocessed_wave_list.add_column('labels', column=label_list)
        preprocessed_wave_list = preprocessed_wave_list.remove_columns(['wave'])
        preprocessed_wave_list = preprocessed_wave_list.with_format('torch')
        return preprocessed_wave_list

    def make_labeled_WavUnimodal_dataset_stepBystep(self, save_target_path, wav_folder_path,
                                                   labeled_json_file_path, preprocessor):
        global WAVE_MAX_LENGTH
        import os, re
        idxes_200 = [[x, x+200] for x in range(0, len(os.listdir(wav_folder_path)), 200)]
        idxes_200[-1] = [idxes_200[-1][0], len(os.listdir(wav_folder_path))-1]
        wav_folder_idxes = [int(re.sub('[^0-9]','',x)) for x in os.listdir(wav_folder_path)]
        wav_folder_idxes.sort()

        print('\nExtracting Features...')
        from tqdm import tqdm
        from datasets.utils.logging import disable_progress_bar, enable_progress_bar
        for n, idxes in enumerate(tqdm(idxes_200)):
            disable_progress_bar()
            wave_list, label_list = self.prepare_wav_unimodal_labeled(
                wav_folder_path=wav_folder_path, labeled_json_file_path=labeled_json_file_path
                , start_idx=wav_folder_idxes[idxes[0]], end_idx=wav_folder_idxes[idxes[1]])
            from datasets import Dataset
            temp_dataset = Dataset.from_dict({'wave':wave_list})

            def preprocess(row):
                return preprocessor(row['wave'], truncation=True, padding=True, max_length=16000*10
                                    , sampling_rate=16000)

            preprocessed_wave_list = temp_dataset.map(preprocess, batched=True, batch_size=100, num_proc=4)
            preprocessed_wave_list = preprocessed_wave_list.add_column('labels', column=label_list)
            preprocessed_wave_list = preprocessed_wave_list.remove_columns(['wave'])
            preprocessed_wave_list = preprocessed_wave_list.with_format('torch')
            preprocessed_wave_list.save_to_disk(f'{save_target_path}/{n}')
        enable_progress_bar()

    def prepare_text_unimodal_labeled(self, labeled_json_file_path, debug=False):
        import os, re, random

        if debug is True: train_data_len = 1200
        else : train_data_len = 4400

        idxes = [int(re.sub('[^0-9]','',x)) for x in os.listdir(labeled_json_file_path)]
        idxes.sort()
        random.shuffle(idxes)

        import json
        json_files = [f'{labeled_json_file_path}/clip_{idx}.json' for idx in idxes]
        text_list=[]
        label_list=[]
        test_text_list=[]
        test_label_list=[]

        from tqdm import tqdm
        for json_file in tqdm(json_files[:train_data_len]):
            if os.path.isfile(json_file) is False : continue

            with open(json_file,'r',encoding='utf-8') as f:
                json_instance = json.load(f)

            for sentence in json_instance:
                text = sentence['text']
                label = self.label_map[sentence['emotion']['text']['emotion']]
                if label == -1:
                    continue
                if len(text) == 0:
                    continue
                text_list.append(text)
                label_list.append(label)

        for json_file in tqdm(json_files[train_data_len:]):
            if os.path.isfile(json_file) is False : continue

            with open(json_file,'r',encoding='utf-8') as f:
                json_instance = json.load(f)

            for sentence in json_instance:
                text = sentence['text']
                label = self.label_map[sentence['emotion']['text']['emotion']]
                if label == -1:
                    continue
                if len(text) == 0:
                    continue
                test_text_list.append(text)
                test_label_list.append(label)

        return text_list, label_list, test_text_list, test_label_list

    def make_labeled_TextUnimodal_dataset(self, labeled_json_file_path, preprocessor, debug=False):
        print('\nReading the data...')
        text_list, label_list, test_text_list, test_label_list = \
            self.prepare_text_unimodal_labeled(labeled_json_file_path=labeled_json_file_path, debug=debug)

        from datasets import Dataset
        temp_dataset = Dataset.from_dict({'text':text_list})
        temp_dataset_test = Dataset.from_dict({'text': test_text_list})

        def preprocess(row):
            return preprocessor(row['text'], max_length=128, padding=True, truncation=True)

        print('\nTokenizing data...')
        preprocessed_dataset = temp_dataset.map(preprocess, batched=True, batch_size=1)
        preprocessed_dataset_test = temp_dataset_test.map(preprocess, batched=True, batch_size=1)
        preprocessed_dataset = preprocessed_dataset.add_column('labels', column=label_list)
        preprocessed_dataset = preprocessed_dataset.remove_columns(['text'])
        preprocessed_dataset = preprocessed_dataset.with_format('torch')
        preprocessed_dataset_test = preprocessed_dataset_test.add_column('labels', column=test_label_list)
        preprocessed_dataset_test = preprocessed_dataset_test.remove_columns(['text'])
        preprocessed_dataset_test = preprocessed_dataset_test.with_format('torch')
        return preprocessed_dataset, preprocessed_dataset_test


class PrepareMultimodalDataset:
    def __init__(self, label_map):
        self.label_map = label_map

    def prepare_Multimodal_labeled(self, wav_folder_path, labeled_json_file_path, cut_arousal=True, start_idx=-1, end_idx=-1):
        import os
        if start_idx != -1 and end_idx != -1:
            idxes = [str(x) for x in range(int(start_idx), int(end_idx))]
        else:
            idxes = [str(x) for x in range(1, len(os.listdir(wav_folder_path)) + 1)]

        import json
        wav_files = [f'{wav_folder_path}/clip_{idx}.wav' for idx in idxes]
        json_files = [f'{labeled_json_file_path}/clip_{idx}.json' for idx in idxes]
        wave_list = []
        label_list = []
        text_list = []

        for wav_file, json_file in zip(wav_files, json_files):
            if os.path.isfile(wav_file) is False: continue

            wave_instance, _ = sf.read(wav_file)
            with open(json_file, 'r', encoding='utf-8') as f:
                json_instance = json.load(f)

                for sentence in json_instance:
                    start = round(sentence['start_wav_time'])
                    end = round(sentence['end_wav_time'])
                    label = self.label_map[sentence['emotion']['multimodal']['emotion']]

                    if cut_arousal is True:
                        if sentence['emotion']['multimodal']['valence'] >= 3 and label==2:
                            continue
                        elif sentence['emotion']['multimodal']['valence'] <= 6 and label==1:
                            continue

                    text = sentence['text']
                    if label == -1:
                        continue
                    if len(wave_instance[start:end]) == 0:
                        print('wave split error!')
                        continue
                    text_list.append(text)
                    wave_list.append(wave_instance[start:end])
                    label_list.append(label)

        return wave_list, text_list, label_list

    def make_labeled_Multimodal_dataset_stepBystep(self, save_target_path, wav_folder_path,labeled_json_file_path,
                                                   text_preprocessor, wave_preprocessor, maxlen=10):
        print('\nReading the data...')
        global WAVE_MAX_LENGTH
        import os, re
        idxes_200 = [[x, x+200] for x in range(0, len(os.listdir(wav_folder_path)), 200)]
        idxes_200[-1] = [idxes_200[-1][0], len(os.listdir(wav_folder_path))-1]
        wav_folder_idxes = [int(re.sub('[^0-9]','',x)) for x in os.listdir(wav_folder_path)]
        wav_folder_idxes.sort()

        print('\nExtracting Features and Tokenizing...')
        from tqdm import tqdm
        from datasets.utils.logging import disable_progress_bar, enable_progress_bar
        from datasets import Dataset, concatenate_datasets
        for n, idxes in enumerate(tqdm(idxes_200)):
            disable_progress_bar()
            wave_list, text_list, label_list = self.prepare_Multimodal_labeled(
                wav_folder_path=wav_folder_path, labeled_json_file_path=labeled_json_file_path
                , start_idx=wav_folder_idxes[idxes[0]], end_idx=wav_folder_idxes[idxes[1]])
            temp_wave_dataset = Dataset.from_dict({'wave': wave_list})
            temp_text_dataset = Dataset.from_dict({'text': text_list})

            def preprocess_wave(row):
                return wave_preprocessor(row['wave'], truncation=True, padding=True,
                                         max_length=16000*maxlen, sampling_rate=16000)

            preprocessed_wave_dataset = temp_wave_dataset.map(preprocess_wave, batched=True, batch_size=100, num_proc=4)
            preprocessed_wave_dataset = preprocessed_wave_dataset.rename_column('input_values', 'audio_input_values')
            preprocessed_wave_dataset = preprocessed_wave_dataset.rename_column('attention_mask', 'audio_attention_mask')

            def preprocess_text(row):
                return text_preprocessor(row['text'], max_length=128, padding=True, truncation=True)

            preprocessed_text_dataset = temp_text_dataset.map(preprocess_text, batched=True, batch_size=1)
            preprocessed_text_dataset = preprocessed_text_dataset.rename_column('input_ids','text_input_ids')
            preprocessed_text_dataset = preprocessed_text_dataset.rename_column('attention_mask', 'text_attention_mask')
            preprocessed_text_dataset = preprocessed_text_dataset.rename_column('token_type_ids', 'text_token_type_ids')

            preprocessed_dataset = concatenate_datasets([preprocessed_wave_dataset, preprocessed_text_dataset], axis=1)
            preprocessed_dataset = preprocessed_dataset.add_column('labels', column=label_list)
            preprocessed_dataset = preprocessed_dataset.remove_columns(['wave','text'])
            preprocessed_dataset = preprocessed_dataset.with_format('torch')
            preprocessed_dataset.save_to_disk(f'{save_target_path}/{n}')
        enable_progress_bar()

    def make_labeled_Multimodal_dataset_stepBystep_final(self, save_target_path, wav_folder_path,labeled_json_file_path):
        print('\nReading the data...')
        global WAVE_MAX_LENGTH
        import os, re

        idxes_200 = [[x, x+200] for x in range(0, len(os.listdir(wav_folder_path)), 200)]
        idxes_200[-1] = [idxes_200[-1][0], len(os.listdir(wav_folder_path))-1]
        wav_folder_idxes = [int(re.sub('[^0-9]','',x)) for x in os.listdir(wav_folder_path)]
        wav_folder_idxes.sort()

        print('\nExtracting Features and Tokenizing...')
        from tqdm import tqdm
        from datasets.utils.logging import disable_progress_bar, enable_progress_bar
        from datasets import Dataset, concatenate_datasets
        for n, idxes in enumerate(tqdm(idxes_200)):
            disable_progress_bar()
            wave_list, text_list, label_list = self.prepare_Multimodal_labeled(
                wav_folder_path=wav_folder_path, labeled_json_file_path=labeled_json_file_path
                , start_idx=wav_folder_idxes[idxes[0]], end_idx=wav_folder_idxes[idxes[1]])

            preprocessed_dataset = Dataset.from_dict({'wave': wave_list, 'text': text_list, 'labels':label_list})
            preprocessed_dataset.save_to_disk(f'{save_target_path}/{n}')
        enable_progress_bar()



if __name__ == '__main__':
    label_map = {'happy': 1, 'surprise': -1, 'angry': 2, 'sad': 2, 'dislike': 2, 'fear': 2, 'contempt': 2, 'neutral': 0}
    prepare = PrepareMultimodalDataset(label_map=label_map)
    prepare.make_labeled_Multimodal_dataset_stepBystep_final(save_target_path='d:/lotte-final-not-termed',
                                                             wav_folder_path='d:/lotte/wav_extracted',
                                                             labeled_json_file_path='d:/lotte/script_extracted')
    prepare.make_labeled_Multimodal_dataset_stepBystep_final(save_target_path='d:/lotte-final-termed',
                                                             wav_folder_path='d:/lotte/wav_extracted',
                                                             labeled_json_file_path='d:/lotte/script_extracted_V_termed')