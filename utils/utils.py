import json
from tqdm import tqdm
import ffmpeg
import numpy as np

import time
WAVE_MAX_LENGTH = 160000
TEXT_MAX_LENGTH = 128

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"'{func.__name__}' took {end_time - start_time} seconds to run.")
        return result
    return wrapper

def align_wav_and_script(json_file, vid_framerate, wav_bitrate, give_term_back_and_forth=0):
    text_avail_keys = []
    split_text_dict = []
    text_obj_set = []

    for json_key in json_file['data'].keys():
        objs = json_file['data'][json_key].keys()
        for obj in objs:
            obj_json = json_file['data'][json_key][obj]
            if obj_json.get('text', False) is not False and obj_json['text'] not in text_obj_set:
                start_wav_time = int(obj_json['text']['script_start']) / vid_framerate
                start_wav_time *= wav_bitrate
                start_wav_time -= give_term_back_and_forth
                if start_wav_time < 0: start_wav_time = 0

                end_wav_time = int(obj_json['text']['script_end']) / vid_framerate
                end_wav_time *= wav_bitrate
                end_wav_time += give_term_back_and_forth

                split_text_dict.append({
                    'text': obj_json['text']['script'],
                    'start_frame': obj_json['text']['script_start'],
                    'end_frame': obj_json['text']['script_end'],
                    'start_wav_time': start_wav_time,
                    'end_wav_time': end_wav_time,
                    'emotion': obj_json['emotion']
                })
                text_obj_set.append(obj_json['text'])
                text_avail_keys.append(json_key)
    return split_text_dict


def convert_mp4_to_wav(file_path, output_path, mono=True):
    framerate = ffmpeg.probe(file_path)['streams'][0]['r_frame_rate']
    audio_bitrate = '16000'
    stream = ffmpeg.input(file_path)
    if mono is True:
        stream = ffmpeg.output(stream.audio, output_path, loglevel="quiet", **{'ar': audio_bitrate, 'ac': 1})
    else:
        stream = ffmpeg.output(stream.audio, output_path, loglevel="quiet", **{'ar':audio_bitrate})
    ffmpeg.run(stream, overwrite_output=True)

    framerate = framerate.split('/')
    framerate = np.float32(framerate[0]) / np.float32(framerate[1])
    return framerate, int(audio_bitrate)

@timing_decorator
def make_labeled_json_file(output_dir, mp4_input_dir = "d:/lotte/given_dataset"
                           , wav_folder_path = 'd:/lotte/wav_extracted', give_term_back_and_forth=0):
    from tqdm import tqdm
    import json
    import os
    import soundfile as sf

    if os.path.exists(output_dir) is True:
        print('making script already done!')
        return

    os.makedirs(output_dir)
    folder_list = os.listdir(mp4_input_dir)
    folder_list = [x for x in folder_list if 'zip' not in x]

    for folder in folder_list:
        clip_folders = [x for x in os.listdir(f'{mp4_input_dir}/{folder}/{folder}')]

        for clip_folder in tqdm(clip_folders):
            mp4_file_path = f'{mp4_input_dir}/{folder}/{folder}/{clip_folder}/{clip_folder}.mp4'
            json_file_path = f'{mp4_input_dir}/{folder}/{folder}/{clip_folder}/{clip_folder}.json'

            try:
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    json_temp = json.load(f)
            except UnicodeDecodeError:
                with open(json_file_path, 'r') as f:
                    json_temp = json.load(f)

            vid_framerate, wav_bitrate = convert_mp4_to_wav(mp4_file_path, f'{wav_folder_path}/{clip_folder}.wav',
                                                            mono=True)

            wav_file, _ = sf.read(f'{wav_folder_path}/{clip_folder}.wav')
            res = align_wav_and_script(json_temp, vid_framerate, wav_bitrate, give_term_back_and_forth)

            with open(f'{output_dir}/{clip_folder}.json','w', encoding='utf-8') as f:
                json.dump(res, f, indent=4, ensure_ascii=False)

import torch
import torch.nn.functional as F
def pad_wave_sequences(batch, max_length=None, padding_value=0.0):
    in_batch_max_length = max(len(item['attention_mask']) for item in batch)
    if max_length > in_batch_max_length:
        batch_to_max = False
    else:
        batch_to_max = True

    input_values = []
    attention_masks = []
    for item in batch:
        in_batch_len = len(item['attention_mask'])
        if batch_to_max is True:
            input_value = F.pad(item['input_values'], (0, max_length - in_batch_len), 'constant', padding_value)
            attention_mask = F.pad(item['attention_mask'], (0, max_length - in_batch_len), 'constant', padding_value)
        else:
            input_value = F.pad(item['input_values'], (0, in_batch_max_length - in_batch_len), 'constant', padding_value)
            attention_mask = F.pad(item['attention_mask'], (0, in_batch_max_length - in_batch_len), 'constant', padding_value)
        input_values.append(input_value)
        attention_masks.append(attention_mask)
    return {'input_values': torch.stack(input_values), 'attention_mask': torch.stack(attention_masks)}


def pad_multimodal_sequences(batch, wave_max_length=None, text_max_length=None, padding_value=0.0):
    in_batch_wave_max_length = max(len(item['audio_attention_mask']) for item in batch)
    if wave_max_length > in_batch_wave_max_length:
        wave_batch_to_max = False
    else:
        wave_batch_to_max = True

    in_batch_text_max_length = max(len(item['text_attention_mask']) for item in batch)
    if text_max_length > in_batch_text_max_length:
        text_batch_to_max = False
    else:
        text_batch_to_max = True

    text_input_ids_list = []
    text_attention_mask_list  = []
    text_token_type_ids_list  = []
    audio_input_values_list  = []
    audio_attention_mask_list  = []

    for item in batch:
        in_batch_wave_len = len(item['audio_attention_mask'])
        in_batch_text_len = len(item['text_attention_mask'])
        if wave_batch_to_max is True:
            audio_input_values = F.pad(item['audio_input_values'], (0, wave_max_length - in_batch_wave_len), 'constant', padding_value)
            audio_attention_mask = F.pad(item['audio_attention_mask'], (0, wave_max_length - in_batch_wave_len), 'constant', padding_value)
        else:
            audio_input_values = F.pad(item['audio_input_values'], (0, in_batch_wave_max_length - in_batch_wave_len), 'constant', padding_value)
            audio_attention_mask = F.pad(item['audio_attention_mask'], (0, in_batch_wave_max_length - in_batch_wave_len), 'constant', padding_value)
        if text_batch_to_max is True:
            text_input_ids = F.pad(item['text_input_ids'], (0, text_max_length - in_batch_text_len), 'constant', padding_value)
            text_attention_mask = F.pad(item['text_attention_mask'], (0, text_max_length - in_batch_text_len), 'constant', padding_value)
            text_token_type_ids = F.pad(item['text_token_type_ids'], (0, text_max_length - in_batch_text_len), 'constant', padding_value)
        else:
            text_input_ids = F.pad(item['text_input_ids'], (0, in_batch_text_max_length - in_batch_text_len), 'constant', padding_value)
            text_attention_mask = F.pad(item['text_attention_mask'], (0, in_batch_text_max_length - in_batch_text_len), 'constant', padding_value)
            text_token_type_ids = F.pad(item['text_token_type_ids'], (0, in_batch_text_max_length - in_batch_text_len), 'constant', padding_value)

        text_input_ids_list.append(text_input_ids)
        text_attention_mask_list.append(text_attention_mask)
        text_token_type_ids_list.append(text_token_type_ids)
        audio_input_values_list.append(audio_input_values)
        audio_attention_mask_list.append(audio_attention_mask)

    return {'text_input_ids': torch.stack(text_input_ids_list),
            'text_attention_mask': torch.stack(text_attention_mask_list),
            'text_token_type_ids': torch.stack(text_token_type_ids_list),
            'audio_input_values': torch.stack(audio_input_values_list),
            'audio_attention_mask': torch.stack(audio_attention_mask_list)}


def wave_collate_fn(batch):
    global WAVE_MAX_LENGTH
    with torch.no_grad():
        labels = [x.pop('labels') for x in batch]
        inputs = pad_wave_sequences(batch, max_length=WAVE_MAX_LENGTH)
        inputs['labels'] = torch.tensor(labels)
    return inputs


def multimodal_collate_fn(batch):
    global WAVE_MAX_LENGTH, TEXT_MAX_LENGTH
    with torch.no_grad():
        if batch[0].get('labels', False) is False:
            return pad_multimodal_sequences(batch, wave_max_length=WAVE_MAX_LENGTH, text_max_length=TEXT_MAX_LENGTH)
        else:
            labels = [x.pop('labels') for x in batch]
            inputs = pad_multimodal_sequences(batch, wave_max_length=WAVE_MAX_LENGTH, text_max_length=TEXT_MAX_LENGTH)
            inputs['labels'] = torch.tensor(labels)
    return inputs


# def seed_trial_pipeline(func):
#     def wrapper(*args, **kwargs):
#         from transformers import set_seed
#         res_metrics = 0
#
#         for seed in seed_list:
#             set_seed(seed)
#             res_metrics += func(*args, **kwargs)
#
#         res_metrics /= len(seed_list)
#         with open(target_directory, 'w') as f:
#             json.dump(res_metrics, f, indent=4)
if __name__ == '__main__':
    make_labeled_json_file(output_dir='d:/lotte/script_extracted')