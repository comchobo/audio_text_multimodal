import json
from transformers import set_seed
from .training_codes.train_models import train_CA_multimodal_model, train_simple_multimodal_model, \
    train_ys_multimodal_model, train_wave_unimodal_model, train_text_unimodal_model

if __name__ == '__main__':
    wav_folder_path = 'd:/lotte/wav_extracted'
    output_labeled_json_file_path = 'd:/lotte/script_extracted_V_termed'

    total_metrics = {
        #
        # 'CA_multimodal_safe': {
        #     'func_name': 'train_CA_multimodal_model', 'param_size': 0, 'score': 0,
        #     'data_folder_path': 'd:/lotte/labeled_dataset_multimodal_V_termed_safe',
        #     'arguments_path': './data/arguments/experimental_multimodal_cw.json'},

        'simple_multimodal_cw_safe': {
            'func_name': 'train_simple_multimodal_model', 'param_size': 0, 'score': 0,
            'data_folder_path': 'data/labeled_dataset_multimodal_V_termed_cw_safe',
            'arguments_path': './data/arguments/simple_multimodal_trial_cw.json'}
        # 'wave_unimodal': {
        #     'func_name': 'train_wave_unimodal_model', 'param_size': 0, 'score': 0,
        #     'data_folder_path': 'd:/lotte/wave_labeled_dataset_max10',
        #     'arguments_path': './data/arguments/uniwave_trial.json'},
        # 'text_unimodal': {
        #     'func_name': 'train_text_unimodal_model', 'param_size': 0, 'score': 0,
        #     'data_folder_path': 'd:/lotte/labeled_dataset_unimodal_V_termed',
        #     'arguments_path': './data/arguments/unitext_trial.json'},

    }

    # lower the lr or enhance bsz
    set_seed(44)

    for key in total_metrics.keys():
        print(f"\nTraining {total_metrics[key]['func_name']} now...\n")
        total_metrics[key]['score']=locals()[total_metrics[key]['func_name']](
            preprocessed_data_folder_path=total_metrics[key]['data_folder_path'],
            labeled_json_file_path=output_labeled_json_file_path,
            wav_folder_path=wav_folder_path,
            training_arguments_file_path=total_metrics[key]['arguments_path'])
        with open(f'./results_{key}.json', 'w') as f:
            json.dump(total_metrics[key], f, indent=4)
