def train_text_unimodal_model(labeled_json_file_path, label_map,
                              training_arguments_file_path=None, debug=True,
                              preprocessed_data_folder_path=None, wav_folder_path=None):
    from Modules.Train import Text_Unimodal_trainer
    TextUnimodalTrainer = Text_Unimodal_trainer(debug=debug)
    TextUnimodalTrainer.load_and_preprocess_dataset(save_target_path=''
                                                    , labeled_json_file_path=labeled_json_file_path, label_map=label_map)
    TextUnimodalTrainer.set_arguments(training_arguments_file_path)
    return TextUnimodalTrainer.train()


def train_wave_unimodal_model(preprocessed_data_folder_path, labeled_json_file_path, wav_folder_path, label_map,
                              training_arguments_file_path=None, debug=True, do_normalize=True,
                              WAVE_MAX_LENGTH=16000*10):
    from Modules.Train import Wave_Unimodal_trainer
    WaveUnimodalTrainer = Wave_Unimodal_trainer(debug=debug, do_normalize=do_normalize, WAVE_MAX_LENGTH=WAVE_MAX_LENGTH)
    WaveUnimodalTrainer.preprocess_dataset(save_target_path=preprocessed_data_folder_path,
                                           wav_folder_path=wav_folder_path,
                                           labeled_json_file_path=labeled_json_file_path,
                                           label_map=label_map)

    WaveUnimodalTrainer.load_preprocessed_dataset(preprocessed_data_folder_path)
    WaveUnimodalTrainer.set_arguments(training_arguments_file_path)
    return WaveUnimodalTrainer.train()


def train_simple_multimodal_model(preprocessed_data_folder_path, labeled_json_file_path, wav_folder_path, label_map,
                                  training_arguments_file_path=None, debug=True, do_normalize=False,
                                  WAVE_MAX_LENGTH=16000*10):
    from Modules.Train import Wave_Text_Simple_Multimodal_trainer
    ys_trainer = Wave_Text_Simple_Multimodal_trainer(text_model_path='embargo',
                          wave_model_path='kresnik/wav2vec2-large-xlsr-korean',
                          debug=debug, do_normalize=do_normalize, WAVE_MAX_LENGTH=WAVE_MAX_LENGTH)

    ys_trainer.preprocess_multimodal_dataset(save_target_path=preprocessed_data_folder_path,
                                              wav_folder_path=wav_folder_path,
                                              labeled_json_file_path=labeled_json_file_path,
                                              label_map=label_map)
    ys_trainer.load_preprocessed_dataset(preprocessed_data_folder_path)
    ys_trainer.set_arguments(training_arguments_file_path)
    return ys_trainer.train()


def train_CA_multimodal_model(preprocessed_data_folder_path, labeled_json_file_path, wav_folder_path, label_map,
                              training_arguments_file_path=None,
                              debug=True, do_normalize=True, normalize='traditional',
                              WAVE_MAX_LENGTH=16000*10):
    from Modules.Train import Wave_Text_CrossAttention_Multimodal_trainer
    CrossAttentionMultimodalTrainer = Wave_Text_CrossAttention_Multimodal_trainer(
        text_model_path="kykim/electra-kor-base",
        wave_model_path="kresnik/wav2vec2-large-xlsr-korean",
        debug=debug, do_normalize=do_normalize, normalize=normalize, WAVE_MAX_LENGTH=WAVE_MAX_LENGTH)

    CrossAttentionMultimodalTrainer.preprocess_multimodal_dataset(save_target_path=preprocessed_data_folder_path,
                                                          wav_folder_path=wav_folder_path,
                                                          labeled_json_file_path=labeled_json_file_path,
                                                          label_map=label_map)
    CrossAttentionMultimodalTrainer.load_preprocessed_dataset(preprocessed_data_folder_path)
    CrossAttentionMultimodalTrainer.set_arguments(training_arguments_file_path)
    return CrossAttentionMultimodalTrainer.train()


def train_ys_multimodal_model(preprocessed_data_folder_path, labeled_json_file_path, wav_folder_path, label_map,
                              training_arguments_file_path=None, debug=True, do_normalize=False,
                              WAVE_MAX_LENGTH=16000*10):
    from Modules.Train import ys_Trainer
    ys_trainer = ys_Trainer(text_model_path='embargo',
                          wave_model_path='kresnik/wav2vec2-large-xlsr-korean',
                          debug=debug, do_normalize=do_normalize, WAVE_MAX_LENGTH=WAVE_MAX_LENGTH)

    ys_trainer.preprocess_multimodal_dataset(save_target_path=preprocessed_data_folder_path,
                                              wav_folder_path=wav_folder_path,
                                              labeled_json_file_path=labeled_json_file_path,
                                              label_map=label_map)
    ys_trainer.load_preprocessed_dataset(preprocessed_data_folder_path)
    ys_trainer.set_arguments(training_arguments_file_path)
    return ys_trainer.train()