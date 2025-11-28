import os
import re
import yaml
import numpy as np
import matplotlib.pyplot as plt

import torch
from Models.Solver import Solver
from Models.EHGAMEGAN import Generator
from utils.data_loader import get_loader_segment
from tabulate import tabulate
from sklearn.metrics import auc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
Tools: Register the dataset
Info:  Read datsets in root_path and record key information in dataset.yaml
The format of dataset.yaml:
    - dataset_name:
        - path: xxx/xxx/
        - type: distributed / centralized
        - file_list (distributed):
            - id-1:
                - train: shape
                - test: shape
                - labels: shape
            - id-2:
                - train: shape
                - test: shape
                - labels: shape
        - file_list (centralized):
            - train-id: shape
            - test-id: shape
            - labels-id: shape
        - (TODO)background: the background information of this dataset
'''
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT_PATH = os.path.join(os.path.dirname(CURRENT_PATH), 'data')   # ../data
DEFAULT_YAML_PATH = os.path.join(CURRENT_PATH, 'dataset.yaml')            # ./dataset.yaml


def AutoRegister(root_path: str = DEFAULT_ROOT_PATH, yaml_path: str = DEFAULT_YAML_PATH):
    dataset_map = {}
    for dataset_name in os.listdir(root_path):
        dataset_path = os.path.join(root_path, dataset_name)
        file_list = os.listdir(dataset_path)
        # filter
        file_list = list(filter(lambda x: x.endswith('.npy'), file_list))
        file_list = list(filter(lambda x: not x.startswith(dataset_name), file_list))
        file_list.sort()
        # centralized
        if 'train.npy' in file_list and 'test.npy' in file_list and 'labels.npy' in file_list:
            dataset_map[dataset_name] = {
                'path': dataset_path,
                'type': 'centralized',
                'file_list': {
                    'data': {
                        'train': list(np.load(os.path.join(dataset_path, 'train.npy')).shape),
                        'test': list(np.load(os.path.join(dataset_path, 'test.npy')).shape),
                        'labels': list(np.load(os.path.join(dataset_path, 'labels.npy')).shape),
                    }
                },
                'background': '',
            }
        # dataset with meta_data.yaml
        elif 'meta_data.yaml' in os.listdir(dataset_path):
            meta_info = yaml.safe_load(open(os.path.join(dataset_path, 'meta_data.yaml'), 'r'))
            data_id_list = meta_info['mapping'].keys()
            id_map = {}
            for id_name in data_id_list:
                id_name = str(id_name)
                if id_name not in dataset_map:
                    id_map[id_name] = {}
                if os.path.exists(os.path.join(dataset_path, f'{id_name}.train.npy')):
                    id_map[id_name]['train'] = list(np.load(os.path.join(dataset_path, f'{id_name}.train.npy')).shape)
                elif os.path.exists(os.path.join(dataset_path, f'{id_name}_train.npy')):
                    id_map[id_name]['train'] = list(np.load(os.path.join(dataset_path, f'{id_name}_train.npy')).shape)
                if os.path.exists(os.path.join(dataset_path, f'{id_name}.test.npy')):
                    id_map[id_name]['test'] = list(np.load(os.path.join(dataset_path, f'{id_name}.test.npy')).shape)
                elif os.path.exists(os.path.join(dataset_path, f'{id_name}_test.npy')):
                    id_map[id_name]['test'] = list(np.load(os.path.join(dataset_path, f'{id_name}_test.npy')).shape)
                if os.path.exists(os.path.join(dataset_path, f'{id_name}.labels.npy')):
                    id_map[id_name]['labels'] = list(np.load(os.path.join(dataset_path, f'{id_name}.labels.npy')).shape)
                elif os.path.exists(os.path.join(dataset_path, f'{id_name}_labels.npy')):
                    id_map[id_name]['labels'] = list(np.load(os.path.join(dataset_path, f'{id_name}_labels.npy')).shape)
                dataset_map[dataset_name] = {
                    'path': dataset_path,
                    'type': 'meta_data',
                    'file_list': id_map,
                    'background': '',
                }
        # distributed
        else:
            id_map = {}
            for file_name in file_list:
                id_name = file_name.split('.')[0].split('_')[0]
                if id_name not in id_map:
                    id_map[id_name] = {}
                id_map[id_name][file_name.split('.')[0].split('_')[1]] = list(np.load(os.path.join(dataset_path, file_name)).shape)
            dataset_map[dataset_name] = {
                'path': dataset_path,
                'type': 'distributed',
                'file_list': id_map,
                'background': '',
            }
        # dump
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_map, f)

    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_map, f)


def check_shape(dataset_name: str, root_path: str = DEFAULT_ROOT_PATH):
    dataset_path = os.path.join(root_path, dataset_name)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset {dataset_name} not found in {root_path}")
    files = os.listdir(dataset_path)
    shape_map = {}
    for file in files:
        data = np.load(os.path.join(dataset_path, file))
        shape_map[file] = data.shape
    for i in shape_map:
        print(i, shape_map[i])


class ConvertorBase:
    output_type = 'index'
    def __init__(self, save_path: str):
        self.save_path = save_path
        self.normal_save_path = os.path.join(self.save_path, 'normal')
        self.abnormal_save_path = os.path.join(self.save_path, 'abnormal')
        self.ensure_dir()
    def ensure_dir(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if not os.path.exists(self.normal_save_path):
            os.makedirs(self.normal_save_path)
        if not os.path.exists(self.abnormal_save_path):
            os.makedirs(self.abnormal_save_path)

    def convert_and_save(self, data, name: str):
        return data
    def load(self, idx: int) -> dict:
        res = {
            'type': 'index',
            'data': idx,
        }
        return res


class ImageConvertor(ConvertorBase):
    output_type = 'image'
    def __init__(self, save_path: str):
        super().__init__(save_path)
        self.width = 1500
        self.height = 320
        self.dpi = 100
        self.x_ticks = 100
        self.aux_enable = True
        self.reconstructed_line_color = 'red'
        self.line_color = 'blue'
        self.x_rotation = 90
        plt.rcParams.update({'font.size': 8})
        # convert to inches
        self.figsize = (self.width/self.dpi, self.height/self.dpi)

    def set_config(self, width: int, height: int, dpi: int, x_ticks: int, aux_enable: bool):
        self.width = width
        self.height = height
        self.dpi = dpi
        self.x_ticks = x_ticks
        self.aux_enable = aux_enable
        # convert to inches
        self.figsize = (self.width/self.dpi, self.height/self.dpi)

    def convert_and_save(self, data, name: int, separate: str = '', reconstructed_data=None):
        # Check if the shape of data is correct
        if len(data.shape) > 2:
            raise ValueError(f"Only accept 1D data, but got {(data.shape)}")
        else:
            data_checked = data.reshape(-1)

        # If reconstructed_data is provided, ensure it has the same shape
        if reconstructed_data is not None:
            if len(reconstructed_data.shape) > 2:
                raise ValueError(f"Only accept 1D reconstructed data, but got {(reconstructed_data.shape)}")
            reconstructed_data_checked = reconstructed_data.reshape(-1)
            if len(reconstructed_data_checked) != len(data_checked):
                raise ValueError(f"Reconstructed data length ({len(reconstructed_data_checked)}) does not match data length ({len(data_checked)})")

        # Create a new figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Auxiliary lines
        if self.aux_enable:
            for x in range(0, len(data_checked) + 1, self.x_ticks):
                ax.axvline(x=x, color='lightgray', linestyle='--', linewidth=0.5)

        # Plot the original data
        ax.plot(data_checked, color=self.line_color, label='Original')

        # Plot the reconstructed data if provided
        if reconstructed_data is not None:
            ax.plot(reconstructed_data_checked, color=self.reconstructed_line_color, label='Reconstructed', linestyle='--')

        # Add legend to distinguish original and reconstructed data
        ax.legend()

        # Set x-ticks and limits
        plt.xticks(rotation=self.x_rotation)
        ax.set_xticks(range(0, len(data_checked) + 1, self.x_ticks))
        ax.set_xlim(0, len(data_checked) + 1)

        # Adjust layout and save the figure
        fig.tight_layout(pad=0.1)
        if separate == 'normal':
            fig.savefig(os.path.join(self.normal_save_path, f"{name}.png"), bbox_inches='tight')
        elif separate == 'abnormal':
            fig.savefig(os.path.join(self.abnormal_save_path, f"{name}.png"), bbox_inches='tight')
        else:
            fig.savefig(os.path.join(self.save_path, f"{name}.png"), bbox_inches='tight')
        plt.close()

    def load(self, name: int):
        image_path = os.path.join(self.save_path, f"{name}.png")
        res = {
            'type': 'image',
            'data': image_path
        }
        return res


class TextConvertor(ConvertorBase):
    output_type = 'text'
    def __init__(self, save_path: str):
        super().__init__(save_path)
    def convert_and_save(self, data, name: int, separate: str = '', reconstructed_data=None):
        with open(os.path.join(self.save_path, f"{name}.txt"), 'w') as f:
            # Format and save the original data
            formatted_data = self.format(data)
            f.write("Original Data:\n")
            f.write(formatted_data + "\n")

            # If reconstructed_data is provided, format and save it as well
            if reconstructed_data is not None:
                formatted_reconstructed = self.format(reconstructed_data)
                f.write("\nReconstructed Data:\n")
                f.write(formatted_reconstructed + "\n")

        # Save to normal/abnormal subdirectories if specified
        if separate == 'normal':
            save_path = os.path.join(self.normal_save_path, f"{name}.txt")
        elif separate == 'abnormal':
            save_path = os.path.join(self.abnormal_save_path, f"{name}.txt")
        else:
            save_path = os.path.join(self.save_path, f"{name}.txt")

        with open(save_path, 'w') as f:
            f.write("Original Data:\n")
            f.write(formatted_data + "\n")
            if reconstructed_data is not None:
                f.write("\nReconstructed Data:\n")
                f.write(formatted_reconstructed + "\n")
    def load(self, name: int):
        text_path = os.path.join(self.save_path, f"{name}.txt")
        res = {
            'type': 'text',
            'data': text_path
        }
        return res
    def format(self, data):
        scaled_data = data * 1000
        sclaed_data = scaled_data.astype(int)
        formatted_data = np.array2string(sclaed_data, separator=',').strip('[]')
        splited_data = formatted_data.split(',')
        format_data_list = []
        for number in splited_data:
            if number == '':
                spaced_number = 'Nan'
            else:
                spaced_number = " ".join(list(number))
            format_data_list.append(spaced_number)
        formatted_data = ','.join(format_data_list)
        formatted_data = formatted_data.replace('\n', '')
        return formatted_data


# padding with nan
def padding(array, target_len):
    current_len = array.shape[0]
    channels = 1 if len(array.shape) == 1 else array.shape[1]
    if current_len >= target_len:
        return array
    else:
        padding_len = target_len - current_len
        padding_array = np.full((padding_len, channels), np.nan)
        return np.concatenate((array, padding_array), axis=0)

# remove nan padding
def remove_padding(array):
    if len(array.shape) == 1:
        return array[~np.isnan(array)]
    else:
        return array[~np.isnan(array).all(axis=1)]


class RawDataset:
    def __init__(self, dataset_name: str, sample_rate: float = 1, normalization_enable: bool = True, yaml_path: str = DEFAULT_YAML_PATH) -> None:
        self.dataset_name = dataset_name
        self.yaml_path = yaml_path
        dataset_map = yaml.safe_load(open(self.yaml_path, 'r'))
        if dataset_name not in dataset_map:
            raise FileNotFoundError(f"Dataset {dataset_name} not found in {yaml_path}")
        self.dataset_info = dataset_map[dataset_name]
        self.sample_rate = sample_rate
        self.normalization_enable = normalization_enable

    def get_background_info(self):
        return str(self.dataset_info['background'])

    def ensure_dir(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)

    def sampling(self, data):
        interval = int(1/self.sample_rate)
        return data[::interval]

    def normalize(self, data):
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std

    def separate_label_make(self, dataset_output_dir, id, mode, window_size, stride, convertor_class, image_config: dict = {}, drop_last: bool = True):
        # Define the base data directory
        base_data_dir = self.dataset_info['path']
        print(f"Starting separate_label_make for dataset {self.dataset_name}, id {id}, mode {mode}")
        print(f"Base data directory: {base_data_dir}")
        print(f"Window size: {window_size}, Stride: {stride}")

        # Use the new dataloader to get data
        print("Loading train and test data...")
        dataset_name_for_loader = self.dataset_name  # e.g., 'UCR'
        dataset, train_loader = get_loader_segment(
            base_data_dir, batch_size=32, win_size=window_size, step=stride,
            mode='train', dataset=dataset_name_for_loader, horizon=1
        )
        _, test_loader = get_loader_segment(
            base_data_dir, batch_size=32, win_size=window_size, step=stride,
            mode='test', dataset=dataset_name_for_loader, horizon=1
        )
        print(f"Train loader created with {len(train_loader)} batches")
        print(f"Test loader created with {len(test_loader)} batches")

        # Extract data from the first batch to determine input_c (assuming consistent shape)
        train_data, _ = next(iter(train_loader))
        input_c = train_data.shape[2] if train_data.dim() == 3 else 1
        print(f"Input channels (input_c): {input_c}")

        # EH-GAM-EGAN Configuration
        config = {
            'data_path': self.dataset_info['path'],
            'dataset': self.dataset_name,
            'batch_size': 32,
            'win_size': window_size,
            'input_c': input_c,
            'latent_dim': 100,
            'lr': 0.0002,
            'num_epochs': 50,
            'b1': 0.5,
            'b2': 0.999,
            'model_save_path': './models/eh_gam_egan_model',
            'gat_inter_dim': 256,
            'decay': 0.0,
            'alpha': 0.5,
            'beta': 0.5,
            'mode': 'inference'  # Default mode
        }
        print("EH-GAM-EGAN configuration set")

        # Train or load EH-GAM-EGAN
        solver = Solver(config)
        model_path = f'{config["model_save_path"]}_{self.dataset_name}/model.ckpt'
        if not os.path.exists(model_path):
            print(f"Model checkpoint not found at {model_path}. Training EH-GAM-EGAN for {self.dataset_name}...")
            config['mode'] = 'train'
            solver = Solver(config)
            solver.train_with_loader()
            config['mode'] = 'inference'
            print("Training completed")
        else:
            print(f"Loading pre-trained EH-GAM-EGAN model from {model_path}")

        generator = Generator(win_size=window_size, latent_dim=config['latent_dim'], input_c=input_c).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        generator.load_state_dict(checkpoint['model_G_state_dict'])
        generator.eval()
        print("Generator model loaded and set to evaluation mode")

        # 2. get the save path of the convertor & init the convertor
        convertor_save_path = os.path.join(dataset_output_dir, id, mode, convertor_class.output_type)
        self.ensure_dir(convertor_save_path)
        convertor = convertor_class(save_path=convertor_save_path)
        if image_config != {}:
            convertor.set_config(**image_config)
        print(f"Convertor initialized with save path: {convertor_save_path}")

        # 3. convert & save using test loader
        window_data_save_path = os.path.join(dataset_output_dir, id, mode, 'data.npy')
        window_data_array = []
        reconstructed_data_array = []
        test_labels_array = []
        total_batches = len(test_loader)
        batch_size = config['batch_size']

        print(f"Starting inference on {total_batches} batches from test_loader...")
        if hasattr(dataset, 'test_labels'):
            print(f"Dataset test_labels shape: {dataset.test_labels.shape}")
        else:
            print("No test_labels attribute found in dataset.")

        window_idx = 0  # Global window index across all batches
        for i, (window_data, _) in enumerate(test_loader):
            if i % 10 == 0:
                print(f"Processing batch {i + 1}/{total_batches} ({(i + 1) / total_batches * 100:.1f}%)")

            # Ensure (B, W, C). Your loader gives (B, W) so add a channel dim.
            if window_data.dim() == 2:
                window_data = window_data.unsqueeze(-1)  # (B, W, 1)

            batch_num_windows = window_data.shape[0]

            # Keep a numpy copy for saving later
            window_data_array.append(window_data.cpu().numpy())  # (B, W, C)

            # Work in torch for the model
            input_tensor = window_data.to(device=device, dtype=torch.float32)  # (B, W, C)

            # Sample latent noise for the Generator: (B, latent_dim)
            B = input_tensor.shape[0]
            z = torch.randn(B, config['latent_dim'], device=device, dtype=torch.float32)

            with torch.no_grad():
                gen_out = generator(z)  # (B, C, W)
                reconstructed = gen_out.permute(0, 2, 1)  # -> (B, W, C) for saving
                reconstructed = reconstructed.detach().cpu().numpy()
            reconstructed_data_array.append(reconstructed)

            # Load corresponding labels for each window in the batch
            if hasattr(dataset, 'test_labels'):
                for j in range(batch_num_windows):
                    start_idx = (window_idx + j) * stride
                    end_idx = start_idx + window_size
                    if end_idx > len(dataset.test_labels):
                        # Handle the case where the window extends beyond the dataset length
                        if drop_last:
                            continue  # Skip this window
                        else:
                            # Pad the labels with zeros if necessary
                            labels = np.zeros(window_size, dtype=dataset.test_labels.dtype)
                            valid_length = len(dataset.test_labels) - start_idx
                            if valid_length > 0:
                                labels[:valid_length] = dataset.test_labels[start_idx:end_idx]
                    else:
                        labels = dataset.test_labels[start_idx:end_idx]
                    print(f"Window {window_idx + j}, Labels shape before append: {labels.shape}")  # Debug print
                    # Ensure labels is at least 1D with length window_size
                    if labels.size == 0:
                        print(f"Warning: Empty labels for window {window_idx + j}, skipping.")
                        continue
                    labels = np.atleast_1d(labels)  # Ensure at least 1D
                    if len(labels.shape) > 1:
                        labels = labels.flatten()  # Flatten to 1D if higher-dimensional
                    if labels.shape[0] != window_size:
                        raise ValueError(
                            f"Labels shape {labels.shape} does not match window_size {window_size} "
                            f"for window {window_idx + j} (start_idx={start_idx}, end_idx={end_idx})"
                        )
                    test_labels_array.append(labels)
            window_idx += batch_num_windows

        print("Inference completed. Concatenating arrays...")
        window_data_array = np.concatenate(window_data_array, axis=0)
        reconstructed_data_array = np.concatenate(reconstructed_data_array, axis=0)
        if test_labels_array:
            test_labels_array = np.concatenate(test_labels_array, axis=0)
            print(f"Test labels shape after concatenation: {test_labels_array.shape}")  # Debug print
            # Reshape test_labels_array to (num_strides, window_size)
            num_strides = window_data_array.shape[0]
            expected_length = num_strides * window_size
            if test_labels_array.shape[0] != expected_length:
                raise ValueError(
                    f"Test labels length {test_labels_array.shape[0]} does not match expected length "
                    f"{expected_length} (num_strides={num_strides}, window_size={window_size})"
                )
            test_labels_array = test_labels_array.reshape(num_strides, window_size)
            print(f"Test labels shape after reshaping: {test_labels_array.shape}")  # Debug print
            # Add a singleton channel dimension to make labels 3D
            test_labels_array = test_labels_array[:, :, np.newaxis]  # Shape: (num_strides, window_size, 1)
            print(f"Test labels shape after adding channel dimension: {test_labels_array.shape}")
            window_label_save_path = os.path.join(dataset_output_dir, id, mode, 'labels.npy')
            np.save(window_label_save_path, test_labels_array)
            print(f"Test labels saved to {window_label_save_path}")
        else:
            print("No test labels found. Skipping label saving.")
            test_labels_array = None  # Ensure test_labels_array is None if no labels

        np.save(window_data_save_path, window_data_array)
        print(f"Window data saved to {window_data_save_path}")

        # Process for convertor
        separate_id_list = {'normal': [], 'abnormal': [], 'save_path': convertor.save_path}
        num_windows = window_data_array.shape[0]
        print(f"Processing {num_windows} windows with {input_c} channels each...")
        for i in range(num_windows):
            if i % 100 == 0:  # Print progress every 100 windows
                print(f"Processing window {i+1}/{num_windows} ({(i+1)/num_windows*100:.1f}%)")
            for ch in range(input_c):
                window_data = window_data_array[i]
                reconstructed_data = reconstructed_data_array[i]
                if test_labels_array is not None and test_labels_array[i].sum() == 0:
                    separate_id_list['normal'].append(f'{id}-{i}-{ch}')
                else:
                    separate_id_list['abnormal'].append(f'{id}-{i}-{ch}')
                convertor.convert_and_save(
                    np.pad(window_data[:, ch], (0, window_size - len(window_data[:, ch])), mode='constant'),
                    f'{i}-{ch}',
                    reconstructed_data=np.pad(reconstructed_data[:, ch], (0, window_size - len(reconstructed_data[:, ch])), mode='constant'),
                    separate='normal' if test_labels_array is not None and test_labels_array[i].sum() == 0 else 'abnormal'
                )

        print(f"Conversion and saving completed. Normal samples: {len(separate_id_list['normal'])}, Abnormal samples: {len(separate_id_list['abnormal'])}")

        # Save background info
        background_save_path = os.path.join(dataset_output_dir, 'background.txt')
        with open(background_save_path, 'w') as f:
            f.write(self.get_background_info())
        print(f"Background info saved to {background_save_path}")

        print("separate_label_make completed successfully")
        return separate_id_list

    def make(self, dataset_output_dir, id, mode, window_size, stride, convertor_class, image_config: dict = {}, drop_last: bool = True):
        # check data_path
        if id == 'data':
            data_path = os.path.join(self.dataset_info['path'], f"{mode}.npy")
        else:
            data_path = os.path.join(self.dataset_info['path'], f"{id}_{mode}.npy")

        # 1. sampling & normalization
        data = np.load(data_path)
        data = self.sampling(data)
        if self.normalization_enable:
            data = self.normalize(data)

        # 2. get the save path of the convertor & init the convertor
        convertor_save_path = os.path.join(dataset_output_dir, id, mode, convertor_class.output_type)
        self.ensure_dir(convertor_save_path)
        convertor = convertor_class(save_path=convertor_save_path)
        if image_config != {}:
            convertor.image_config(**image_config)

        # 3. convert & save
        # data .npy format: [num_stride, window_size, data_channels]
        window_data_save_path = os.path.join(dataset_output_dir, id, mode, 'data.npy')
        window_data_array = []
        num_stride = (len(data)-window_size) // stride + 1
        data_channels = 1 if len(data.shape) == 1 else data.shape[1]
        # data
        for i in range(num_stride):
            start = i * stride
            window_data = data[start:start+window_size]
            window_data_array.append(window_data)
            # convert & save
            for ch in range(data_channels):
                convertor.convert_and_save(window_data[:, ch], f'{i}-{ch}')
        if not drop_last:
            start = num_stride * stride
            window_data = data[start:]
            padded_window_data = padding(window_data, window_size)
            window_data_array.append(padded_window_data)
            for ch in range(data_channels):
                convertor.convert_and_save(window_data[:, ch], f'{num_stride}-{ch}')

        window_data_array = np.array(window_data_array)
        np.save(window_data_save_path, window_data_array)
        # label
        window_label_save_path = os.path.join(dataset_output_dir, id, mode, 'labels.npy')
        if mode == 'test':
            if id == 'data':
                labels_path = os.path.join(self.dataset_info['path'], f"labels.npy")
            else:
                labels_path = os.path.join(self.dataset_info['path'], f"{id}_labels.npy")
            # sampling
            labels = np.load(labels_path)
            labels = self.sampling(labels)
            # label .npy format: [num_stride, window_size, label_channels]
            window_label_array = []
            label_channels = 1 if len(labels.shape) == 1 else labels.shape[1]
            for i in range(num_stride):
                start = i * stride
                window_label = labels[start:start+window_size]
                window_label_array.append(window_label)
            if not drop_last:
                start = num_stride * stride
                window_label = labels[start:]
                window_label = padding(window_label, window_size)
                window_label_array.append(window_label)
            window_label_array = np.array(window_label_array)
            # Ensure labels are 3D by adding a singleton channel dimension
            if window_label_array.ndim == 2:
                window_label_array = window_label_array[:, :, np.newaxis]  # Shape: (num_strides, window_size, 1)
            np.save(window_label_save_path, window_label_array)
        # background
        background_save_path = os.path.join(dataset_output_dir, 'background.txt')
        with open(background_save_path, 'w') as f:
            f.write(self.get_background_info())

    '''
    output directory: "output_dir/dataset_name/id/name/convertor_type"
    '''
    def convert_data(self, output_dir: str, mode: str, window_size: int, stride: int, convertor_class: ConvertorBase,
                     image_config: dict = {}, drop_last: bool = True, data_id_list: list = []):
        data_id_list = [] if data_id_list == [''] else data_id_list
        dataset_output_dir = os.path.join(output_dir, self.dataset_name)
        self.ensure_dir(dataset_output_dir)
        if self.dataset_info['type'] == 'centralized' or self.dataset_info['type'] == 'distributed':
            id_list = self.dataset_info['file_list'].keys() if data_id_list == [] else data_id_list
        elif self.dataset_info['type'] == 'meta_data':
            id_list = self.dataset_info['file_list'].keys() if data_id_list == [] else data_id_list
        else:
            raise ValueError(f"Invalid dataset type: {self.dataset_info['type']}")
        structure = {}
        for id in id_list:
            id = str(id)
            id_idx_list = self.separate_label_make(dataset_output_dir, id, mode, window_size, stride, convertor_class, image_config=image_config, drop_last=drop_last)
            structure[id] = id_idx_list
        with open(os.path.join(dataset_output_dir, f"{mode}_structure.yaml"), 'w') as f:
            yaml.dump(structure, f)


DEFAULT_LOG_ROOT = os.path.join(os.path.dirname(CURRENT_PATH), 'log')


class ProcessedDataset:
    def __init__(self, dataset_path: str, mode: str = 'train'):
        self.dataset_path = dataset_path
        self.id_list = list(filter(lambda x: os.path.isdir(os.path.join(dataset_path, x)), os.listdir(dataset_path)))
        self.id_list.sort()
        self.background = open(os.path.join(dataset_path, 'background.txt'), 'r').read()
        self.mode = mode
        self.labels_cache = {}  # Cache for loaded labels to improve performance
        self.count_data_num()

    def get_id_list(self):
        return self.id_list

    def get_instances(self, balanced: bool = True, ratio: float = 0.5, refined: bool = False):
        import yaml
        import random
        structure = yaml.safe_load(open(os.path.join(self.dataset_path, f"{self.mode}_structure.yaml"), 'r'))
        pos = []
        neg = []
        if refined:
            if balanced:
                for id in structure.keys():
                    pos_len = len(structure[id]['abnormal'])
                    neg_len = len(structure[id]['normal'])
                    if neg_len < pos_len:
                        neg += structure[id]['normal']
                        pos += random.sample(structure[id]['abnormal'], neg_len)
                    else:
                        pos += structure[id]['abnormal']
                        neg += random.sample(structure[id]['normal'], pos_len)
                pos = random.sample(pos, int(len(pos)*ratio))
                neg = random.sample(neg, int(len(neg)*ratio))
            else:
                for id in structure.keys():
                    pos += random.sample(structure[id]['abnormal'], int(len(structure[id]['abnormal'])*ratio))
                    neg += random.sample(structure[id]['normal'], int(len(structure[id]['normal'])*ratio))
        else:
            if balanced:
                for id in structure.keys():
                    pos_len = len(structure[id]['abnormal'])
                    neg_len = len(structure[id]['normal'])
                    if neg_len < pos_len:
                        neg += structure[id]['normal']
                        pos += random.sample(structure[id]['abnormal'], neg_len)
                    else:
                        pos += structure[id]['abnormal']
                        neg += random.sample(structure[id]['normal'], pos_len)
            else:
                pos = 0
                neg = 0
                for id in structure.keys():
                    pos += len(structure[id]['abnormal'])
                    neg += len(structure[id]['normal'])
                print(f"Positive: {pos}, Negative: {neg}")
        return pos, neg

    def get_background_info(self):
        if self.background == '':
            self.background = 'No background information'
        return self.background

    def count_data_num(self):
        self.data_id_info = {}
        self.total_data_num = 0
        for data_id in self.id_list:
            item = {
                'num_stride': 0,
                'window_size': 0,
                'data_channels': 0,
                'label_channels': 0,
            }
            label_path = os.path.join(self.dataset_path, data_id, 'test', 'labels.npy')
            data_path = os.path.join(self.dataset_path, data_id, 'test', 'data.npy')
            labels = np.load(label_path)
            data = np.load(data_path)
            print(f"Data ID: {data_id}, Labels shape: {labels.shape}, Data shape: {data.shape}")  # Debug print
            item['num_stride'] = data.shape[0]
            item['window_size'] = data.shape[1]
            item['data_channels'] = data.shape[2]
            # Dynamically determine the number of label channels
            if len(labels.shape) == 3:
                item['label_channels'] = labels.shape[2]
            elif len(labels.shape) == 2:
                item['label_channels'] = 1  # Assume 1 channel if 2D
            elif len(labels.shape) == 1:
                # Reshape 1D labels into 2D: (num_strides, window_size)
                expected_length = item['num_stride'] * item['window_size']
                if labels.shape[0] != expected_length:
                    raise ValueError(
                        f"Label length {labels.shape[0]} does not match expected length "
                        f"{expected_length} (num_strides={item['num_stride']}, window_size={item['window_size']}) "
                        f"for data_id {data_id}"
                    )
                labels = labels.reshape(item['num_stride'], item['window_size'])
                item['label_channels'] = 1  # Assume 1 channel after reshaping
            else:
                raise ValueError(f"Unexpected label shape {labels.shape} for data_id {data_id}")
            self.total_data_num += int(item['num_stride'] * item['data_channels'])
            self.data_id_info[data_id] = item

    def get_total_data_num(self):
        return self.total_data_num

    def get_data_id_info(self, data_id):
        labels = self.load_labels(data_id)
        data = self.load_data(data_id)
        return {
            'num_stride': data.shape[0],
            'data_channels': data.shape[2],
            'label_channels': labels.shape[2],  # Labels are guaranteed to be 3D after load_labels
            'window_size': data.shape[1]
        }

    def load_labels(self, data_id):
        """
        Load the labels for a given data_id from the labels.npy file.
        Ensures that the labels array is 3D by adding a singleton channel dimension if necessary.
        """
        if data_id in self.labels_cache:
            return self.labels_cache[data_id]

        label_path = os.path.join(self.dataset_path, data_id, self.mode, 'labels.npy')
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Labels file not found at {label_path} for data_id {data_id}")

        labels = np.load(label_path)
        print(f"Loaded labels for {data_id} with shape: {labels.shape}")

        # Ensure labels are 3D by adding a singleton channel dimension if necessary
        if labels.ndim == 2:  # Shape: (num_strides, window_size)
            labels = labels[:, :, np.newaxis]  # Shape: (num_strides, window_size, 1)
            print(f"Added singleton channel dimension, new shape: {labels.shape}")
        elif labels.ndim != 3:
            raise ValueError(f"Unexpected label shape {labels.shape} for data_id {data_id}. Expected 2D or 3D array.")

        self.labels_cache[data_id] = labels
        return labels

    def load_data(self, data_id):
        """
        Load the data for a given data_id from the data.npy file.
        """
        data_path = os.path.join(self.dataset_path, data_id, self.mode, 'data.npy')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path} for data_id {data_id}")
        return np.load(data_path)

    def get_label_channel(self, data_id, data_ch):
        """
        Map a data channel to a label channel.
        """
        data_info = self.get_data_id_info(data_id)
        label_channels = data_info['label_channels']
        if label_channels == 1:
            return 0  # Only one label channel, so always return 0
        return data_ch % label_channels  # Example mapping

    # TODO: 后续可以优化读取的次数，提升运行速度
    def get_data(self, data_id, num_stride, ch):
        data_path = os.path.join(self.dataset_path, data_id, self.mode, 'data.npy')
        data = np.load(data_path)
        return remove_padding(data[num_stride, :, ch])

    def get_image(self, data_id, num_stride, ch):
        base_path = os.path.join(self.dataset_path, data_id, self.mode, 'image')
        image_name = f'{num_stride}-{ch}.png'
        # Check normal directory
        normal_path = os.path.join(base_path, 'normal', image_name)
        if os.path.exists(normal_path):
            return normal_path
        # Check abnormal directory
        abnormal_path = os.path.join(base_path, 'abnormal', image_name)
        if os.path.exists(abnormal_path):
            return abnormal_path
        # Fallback to base image directory
        image_path = os.path.join(base_path, image_name)
        if os.path.exists(image_path):
            return image_path
        raise FileNotFoundError(f"Image {image_name} not found in normal, abnormal, or base image directory for data_id {data_id}")

    def get_text(self, data_id, num_stride, ch):
        text_path = os.path.join(self.dataset_path, data_id, self.mode, 'text', f'{num_stride}-{ch}.txt')
        return text_path

    def get_label(self, data_id, num_stride, data_ch):
        data_info = self.get_data_id_info(data_id)
        label_ch = self.get_label_channel(data_id, data_ch)
        labels = self.load_labels(data_id)  # Shape: (num_strides, window_size, label_channels)
        # Since load_labels ensures labels is 3D, we can directly index with label_ch
        return remove_padding(labels[num_stride, :, label_ch])


class EvalDataLoader:
    def __init__(self, dataset_name: str, processed_data_root: str, log_root: str = DEFAULT_LOG_ROOT):
        self.dataset_info = yaml.safe_load(open(DEFAULT_YAML_PATH, 'r'))[dataset_name]
        self.dataset_name = dataset_name
        self.log_root = log_root
        self.log_file_path = os.path.join(log_root, f"{dataset_name}_log.yaml")
        self.dataset_path = os.path.join(processed_data_root, dataset_name)
        self.processed_dataset = ProcessedDataset(self.dataset_path, mode='test')
        self.eval_image_path = os.path.join(log_root, f'{dataset_name}_image')
        if not os.path.exists(self.eval_image_path):
            os.makedirs(self.eval_image_path)
        # load
        self.output_log = yaml.safe_load(open(self.log_file_path, 'r'))
        self.plot_default_config = {
            'width': 1024,
            'height': 320,
            'dpi': 100,
            'x_ticks': 100,
        }

    def label_to_list(self, info: str):
        if info == '[]':
            return []
        else:
            return list(map(int, info.strip('[]').split(',')))

    def abnormal_index_to_range(self, info: str):
        # (start, end)/confidence/abnormal_type
        pattern_range = r'\(\d+,\s\d+\)/\d/[a-z]+'
        pattern_single = r'\(\d+\)/\d/[a-z]+'
        if info == '[]':
            return [[]]
        else:
            abnormal_ranges = re.findall(pattern_range, info)
            range_list = []
            type_list = []
            for range_tuple_confidence in abnormal_ranges:
                range_tuple, confidence, abnormal_type = range_tuple_confidence.split('/')
                range_tuple = range_tuple.strip('()')
                start, end = map(int, range_tuple.split(','))
                confidence = int(confidence)
                type_list.append(abnormal_type)
                range_list.append((start, end, confidence))
            abnomral_singles = re.findall(pattern_single, info)
            for single_point_confidence in abnomral_singles:
                single_point, confidence, abnormal_type = single_point_confidence.split('/')
                single_point = single_point.strip('()')
                confidence = int(confidence)
                single_point = int(single_point)
                type_list.append(abnormal_type)
                range_list.append((single_point, confidence))
            return range_list

    def map_pred_window_index_to_global_index(self, window_index, offset: int = 0):
        global_index_set = set()
        for start_end_confidence in window_index:
            if isinstance(start_end_confidence, tuple) or isinstance(start_end_confidence, list):
                # (A, B, C) or (A, C) or [A, B, C] or [A, C]
                if len(start_end_confidence) == 0:
                    continue
                elif len(start_end_confidence) == 2:
                    point, confidence = start_end_confidence
                    if confidence <= 2:
                        continue
                    # single point
                    global_index_set.add(point+offset)
                elif len(start_end_confidence) == 3:
                    # a range of points
                    start, end, confidence = start_end_confidence
                    if confidence <= 2:
                        continue
                    for i in range(start, end+1):
                        global_index_set.add(i+offset)
                else:
                    raise ValueError(f"Invalid abnormal index format: {start_end_confidence}")
            else:
                global_index_set.add(start_end_confidence+offset)
        return global_index_set

    def map_label_window_index_to_global_index(self, window_index, offset: int = 0):
        global_index_set = set()
        for start_end in window_index:
            if isinstance(start_end, tuple) or isinstance(start_end, list):
                # (A, B) or (A,) or [A] or [A, B]
                if len(start_end) == 0:
                    continue
                elif len(start_end) == 1:
                    # single point
                    global_index_set.add(start_end[0]+offset)
                elif len(start_end) == 2:
                    # a range of points
                    start, end = start_end
                    for i in range(start, end+1):
                        global_index_set.add(i+offset)
                else:
                    raise ValueError(f"Invalid abnormal index format: {start_end}")
            else:
                global_index_set.add(start_end+offset)
        global_index_set = list(global_index_set)
        global_index_set.sort()
        return global_index_set

    def map_pred_window_index_to_global_index_with_confidence(self, window_index, offset: int = 0):
        global_index_set = {confidence: set() for confidence in range(1, 5)}
        for start_end_confidence in window_index:
            if isinstance(start_end_confidence, tuple) or isinstance(start_end_confidence, list):
                # (A, B, C) or (A, C) or [A, B, C] or [A, C]
                if len(start_end_confidence) == 0:
                    continue
                elif len(start_end_confidence) == 2:
                    point, confidence = start_end_confidence
                    # single point
                    if confidence not in global_index_set:
                        continue
                    global_index_set[confidence].add(point+offset)
                elif len(start_end_confidence) == 3:
                    # a range of points
                    start, end, confidence = start_end_confidence
                    for i in range(start, end+1):
                        if confidence not in global_index_set:
                            continue
                        global_index_set[confidence].add(i+offset)
                else:
                    raise ValueError(f"Invalid abnormal index format: {start_end_confidence}")
            else:
                raise ValueError(f"Invalid abnormal index format: {start_end_confidence}")
        for confidence in global_index_set:
            global_index_set[confidence] = list(global_index_set[confidence])
            global_index_set[confidence].sort()
        return global_index_set

    def set_plot_config(self, width: int, height: int, dpi: int, x_ticks: int, aux_enable: bool = False):
        plt.rcParams.update({'font.size': 8})
        self.plot_default_config['width'] = width
        self.plot_default_config['height'] = height
        self.plot_default_config['dpi'] = dpi
        self.plot_default_config['x_ticks'] = x_ticks

    def get_fill_ranges(self, points, continue_thre=1):
        if points == []:
            return []
        start_idx = points[0]
        fill_range_list = []
        for i in range(1, len(points)):
            if points[i] - points[i-1] > continue_thre:
                fill_range_list.append((start_idx, points[i-1]+1))
                start_idx = points[i]
        fill_range_list.append((start_idx, points[-1]+1))
        return fill_range_list

    def plot_figure(self, data, label, pred_points, image_name: str):
        figsize = (self.plot_default_config['width']/self.plot_default_config['dpi'], self.plot_default_config['height']/self.plot_default_config['dpi'])
        fig, ax = plt.subplots(figsize=figsize, dpi=self.plot_default_config['dpi'])
        ax.plot(data, label='data')

        alpha = 0.2
        pred_ranges = self.get_fill_ranges(pred_points)
        for start, end in pred_ranges:
            ax.fill_between(range(start, end), np.min(data), np.max(data), color='green', alpha=alpha, label='pred' if start == pred_ranges[0][0] else '')
        label_points = np.where(label == 1)[0].tolist()
        label_ranges = self.get_fill_ranges(label_points)
        for start, end in label_ranges:
            ax.fill_between(range(start, end), np.min(data), np.max(data), color='red', alpha=alpha, label='label' if start == label_ranges[0][0] else '')

        ax.legend()
        ax.set_xticks(range(0, len(data)+1, self.plot_default_config['x_ticks']))
        ax.set_xlim(0, len(data)+1)
        plt.xticks(rotation=90)
        ax.set_title(image_name)
        fig.tight_layout(w_pad=0.1, h_pad=0)
        plt.savefig(os.path.join(self.eval_image_path, f"{image_name}.png"))
        plt.close()

    def plot_figure_with_confidence(self, data, label, pred_points: dict, image_name: str):
        figsize = (self.plot_default_config['width']/self.plot_default_config['dpi'], self.plot_default_config['height']/self.plot_default_config['dpi'])
        fig, ax = plt.subplots(figsize=figsize, dpi=self.plot_default_config['dpi'])
        ax.plot(data, label='data')

        alpha = 0.2
        color_map = {1: 'gray', 2: 'blue', 3: 'yellow', 4: 'green'}
        for confidence in pred_points:
            pred_ranges = self.get_fill_ranges(pred_points[confidence])
            for start, end in pred_ranges:
                ax.fill_between(range(start, end), np.min(data), np.max(data), color=color_map[confidence], alpha=alpha, label=f'pred(confidence={confidence})' if start == pred_ranges[0][0] else '')
        label_points = np.where(label >= 1)[0].tolist()
        label_ranges = self.get_fill_ranges(label_points)
        for start, end in label_ranges:
            ax.fill_between(range(start, end), np.min(data), np.max(data), color='red', alpha=alpha, label='label' if start == label_ranges[0][0] else '')

        ax.legend()
        ax.set_xticks(range(0, len(data)+1, self.plot_default_config['x_ticks']))
        ax.set_xlim(0, len(data)+1)
        plt.xticks(rotation=90)
        ax.set_title(image_name)
        fig.tight_layout(w_pad=0.1, h_pad=0)
        plt.savefig(os.path.join(self.eval_image_path, f"{image_name}.png"))
        plt.close()

    def adjust_anomaly_detection_results(self, results, labels):
        """
        Adjust anomaly detection results based on the ground-truth labels.
        """
        adjusted_results = results.copy()
        in_anomaly = False
        start_idx = 0

        for i in range(len(labels)):
            if labels[i] == 1 and not in_anomaly:
                in_anomaly = True
                start_idx = i
            elif labels[i] == 0 and in_anomaly:
                in_anomaly = False
                if np.any(results[start_idx:i] == 1):
                    adjusted_results[start_idx:i] = 1

        # Handle the case where the last segment is an anomaly
        if in_anomaly and np.any(results[start_idx:] == 1):
            adjusted_results[start_idx:] = 1
        return adjusted_results

    def eval(self, window_size, stride, vote_thres: int, point_adjust_enable: bool = False, plot_enable: bool = False, channel_shared: bool = False):
        eval_logger = {}
        for data_id in self.output_log:
            data_id_info = self.processed_dataset.get_data_id_info(data_id)
            data_channels = data_id_info['data_channels']
            num_stride = data_id_info['num_stride']
            data_shape = self.dataset_info['file_list'][data_id]['test']
            # [globa_index, channel]
            ch_global_pred_array = np.zeros(data_shape)
            ch_global_label_array = np.zeros(data_shape)
            if data_channels == 1:
                ch_global_pred_array = ch_global_pred_array.reshape(-1, 1)
                ch_global_label_array = ch_global_label_array.reshape(-1, 1)
            for ch in range(data_channels):
                for stride_idx in range(num_stride):
                    if stride_idx not in self.output_log[data_id]:
                        continue
                    if self.output_log[data_id][stride_idx] == {}:
                        continue
                    item = self.output_log[data_id][stride_idx][ch]
                    labels = self.label_to_list(item['labels'])
                    if 'abnormal_index' not in item:
                        item['abnormal_index'] = '[]'
                    item['abnormal_index'] = str(item['abnormal_index'])
                    abnormal_index = self.abnormal_index_to_range(item['abnormal_index'])
                    abnormal_description = item['abnormal_description']
                    image_path = item['image']
                    # map to global index
                    offset = stride_idx * stride
                    abnormal_point_set = self.map_pred_window_index_to_global_index(abnormal_index, offset)
                    label_point_set = self.map_label_window_index_to_global_index(labels, offset)
                    # mark point
                    for label_point in label_point_set:
                        if label_point < ch_global_label_array.shape[0]:    #
                            ch_global_label_array[label_point, ch] = 1
                    for abnormal_point in abnormal_point_set:
                        if abnormal_point < ch_global_pred_array.shape[0]:
                            ch_global_pred_array[abnormal_point, ch] += 1
                    # plot
                    if plot_enable:
                        plot_data = self.processed_dataset.get_data(data_id, stride_idx, ch)
                        plot_label = self.processed_dataset.get_label(data_id, stride_idx, ch)
                        plot_pred = self.map_pred_window_index_to_global_index_with_confidence(abnormal_index, 0)
                        self.plot_figure_with_confidence(plot_data, plot_label, plot_pred, f"{data_id}_{stride_idx}_{ch}")

                # vote in channel
                ch_global_pred_array[:, ch] = (ch_global_pred_array[:, ch] >= vote_thres).astype(int)
                # adjust anomaly detection results
                if point_adjust_enable:
                    ch_global_pred_array[:, ch] = self.adjust_anomaly_detection_results(ch_global_pred_array[:, ch], ch_global_label_array[:, ch])
                # plot
            # count TP, FP, TN, FN
            if channel_shared:
                global_pred_array = np.sum(ch_global_pred_array, axis=1)
                global_label_array = np.sum(ch_global_label_array, axis=1)
                global_pred_array = (global_pred_array >= 1).astype(int)
                global_label_array = (global_label_array >= 1).astype(int)
                eval_logger[data_id] = {
                    'TP': np.sum((global_pred_array == 1) & (global_label_array == 1)),
                    'FP': np.sum((global_pred_array == 1) & (global_label_array == 0)),
                    'TN': np.sum((global_pred_array == 0) & (global_label_array == 0)),
                    'FN': np.sum((global_pred_array == 0) & (global_label_array == 1)),
                }
            else:
                global_pred_array = ch_global_pred_array
                global_label_array = ch_global_label_array
                eval_logger[data_id] = {
                    'TP': 0,
                    'FP': 0,
                    'TN': 0,
                    'FN': 0,
                }
                for ch in range(data_channels):
                    global_pred_array[:, ch] = (global_pred_array[:, ch] >= 1).astype(int)
                    global_label_array[:, ch] = (global_label_array[:, ch] >= 1).astype(int)
                    eval_logger[data_id]['TP'] += np.sum((global_pred_array[:, ch] == 1) & (global_label_array[:, ch] == 1))
                    eval_logger[data_id]['FP'] += np.sum((global_pred_array[:, ch] == 1) & (global_label_array[:, ch] == 0))
                    eval_logger[data_id]['TN'] += np.sum((global_pred_array[:, ch] == 0) & (global_label_array[:, ch] == 0))
                    eval_logger[data_id]['FN'] += np.sum((global_pred_array[:, ch] == 0) & (global_label_array[:, ch] == 1))
                # mean
                eval_logger[data_id]['TP'] /= data_channels
                eval_logger[data_id]['FP'] /= data_channels
                eval_logger[data_id]['TN'] /= data_channels
                eval_logger[data_id]['FN'] /= data_channels

        # all metrics
        TP = sum([eval_logger[data_id]['TP'] for data_id in eval_logger])
        FP = sum([eval_logger[data_id]['FP'] for data_id in eval_logger])
        TN = sum([eval_logger[data_id]['TN'] for data_id in eval_logger])
        FN = sum([eval_logger[data_id]['FN'] for data_id in eval_logger])
        accuracy = (TP + TN) / (TP + FP + TN + FN)
        precision = TP / (TP + FP) if TP + FP != 0 else 0
        recall = TP / (TP + FN) if TP + FN != 0 else 0
        F1_score = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
        print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
        print(f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1_score: {F1_score:.3f}")


class Evaluator:
    def __init__(self, dataset_name: str, stride_length: int, processed_data_root: str, log_root: str = DEFAULT_LOG_ROOT, processed_path_name: str = ''):
        self.dataset_name = dataset_name
        self.stride_length = stride_length
        self.processed_data_root = processed_data_root
        self.log_root = log_root
        self.dataset_info = yaml.safe_load(open(DEFAULT_YAML_PATH, 'r'))[dataset_name]
        if processed_path_name == '':
            name = dataset_name
        else:
            name = processed_path_name
        self.processed_dataset = ProcessedDataset(os.path.join(processed_data_root, name), mode='test')

        log_file_path = os.path.join(log_root, f"{dataset_name}_log.yaml")
        self.load_log_file(log_file_path)

    def get_ranges_from_points(self, point_list: list, continue_thres: int = 1):
        if point_list == []:
            return []
        start_idx = point_list[0]
        ranges_list = []
        for i in range(1, len(point_list)):
            if point_list[i] - point_list[i-1] > continue_thres:
                ranges_list.append((start_idx, point_list[i-1]+1))
                start_idx = point_list[i]
        ranges_list.append((start_idx, point_list[-1]+1))
        return ranges_list

    def decode_label(self, label: str):
        if label == '[]':
            return []
        else:
            label_point_list = list(map(int, label.strip('[]').split(',')))
            return label_point_list

    def decode_abnormal_prediction(self, log_item: dict, channel: int, offset: int = 0):
        pattern_range = r'\(\d+,\s\d+\)/\d/[a-z]+'
        pattern_single = r'\(\d+\)/\d/[a-z]+'
        abnormal_index = str(log_item['abnormal_index'])
        double_check_index = log_item.get('double_check', {'corrected_abnormal_index': '[]'})
        if 'fixed_abnormal_index' in double_check_index:
            corrected_abnormal_index = double_check_index['fixed_abnormal_index']
        elif 'corrected_abnormal_index' in double_check_index:
            corrected_abnormal_index = double_check_index['corrected_abnormal_index']
        else:
            corrected_abnormal_index = '[]'
        output_dict = {
            'prediction': [],
            'double_check': [],
        }

        # prediction of ranges
        abnormal_ranges = re.findall(pattern_range, abnormal_index)
        for item in abnormal_ranges:
            range_tuple, confidence, abnormal_type = item.split('/')
            range_tuple = range_tuple.strip('()')
            start, end = map(int, range_tuple.split(','))
            confidence = int(confidence)
            pred = {
                'channel': channel,
                'start': start+offset,
                'end': end+offset,
                'confidence': confidence,
                'type': abnormal_type,
            }
            output_dict['prediction'].append(pred)
        # prediction of single points
        abnormal_singles = re.findall(pattern_single, abnormal_index)
        for item in abnormal_singles:
            single_point, confidence, abnormal_type = item.split('/')
            single_point = single_point.strip('()')
            confidence = int(confidence)
            single_point = int(single_point)
            pred = {
                'channel': channel,
                'start': single_point+offset,
                'end': single_point+1+offset,  # range(single_point, single_point+1) == [single_point]
                'confidence': confidence,
                'type': abnormal_type,
            }
            output_dict['prediction'].append(pred)
        # double check of ranges
        double_check_ranges = re.findall(pattern_range, corrected_abnormal_index)
        for item in double_check_ranges:
            range_tuple, confidence, abnormal_type = item.split('/')
            range_tuple = range_tuple.strip('()')
            start, end = map(int, range_tuple.split(','))
            confidence = int(confidence)
            pred = {
                'channel': channel,
                'start': start+offset,
                'end': end+offset,
                'confidence': confidence,
                'type': abnormal_type,
            }
            output_dict['double_check'].append(pred)
        # double check of single points
        double_check_singles = re.findall(pattern_single, corrected_abnormal_index)
        for item in double_check_singles:
            single_point, confidence, abnormal_type = item.split('/')
            single_point = single_point.strip('()')
            confidence = int(confidence)
            single_point = int(single_point)
            pred = {
                'channel': channel,
                'start': single_point+offset,
                'end': single_point+1+offset,
                'confidence': confidence,
                'type': abnormal_type,
            }
            output_dict['double_check'].append(pred)
        return output_dict

    def load_log_file(self, log_file_path: str):
        self.raw_log = yaml.safe_load(open(log_file_path, 'r'))
        self.parsed_log = {}
        for data_id in self.raw_log:
            data_id_info = self.processed_dataset.get_data_id_info(data_id)
            data_channels = data_id_info['data_channels']
            num_stride = data_id_info['num_stride']
            data_shape = self.dataset_info['file_list'][data_id]['test']
            # [globa_index, channel]
            self.parsed_log[data_id] = {}
            self.parsed_log[data_id]['raw_data'] = np.zeros(data_shape).reshape(-1, data_channels)
            self.parsed_log[data_id]['label'] = np.zeros(data_shape).reshape(-1, data_channels)
            self.parsed_log[data_id]['prediction'] = []
            self.parsed_log[data_id]['double_check'] = []
            for ch in range(data_channels):
                for stride_idx in range(num_stride):
                    offset = int(stride_idx * self.stride_length)
                    if stride_idx not in self.raw_log[data_id]:
                        continue
                    log_item = self.raw_log[data_id][stride_idx][ch]
                    # raw
                    raw_label = self.decode_label(log_item['labels'])
                    raw_data = self.processed_dataset.get_data(data_id, stride_idx, ch)
                    # map to global
                    for point in raw_label:
                        self.parsed_log[data_id]['label'][point+offset, ch] = 1
                    self.parsed_log[data_id]['raw_data'][offset:offset+len(raw_data), ch] = raw_data
                    # get prediction and double_check
                    parsed_dict = self.decode_abnormal_prediction(log_item, ch, offset)
                    prediction_list = parsed_dict['prediction']
                    double_check_list = parsed_dict['double_check']
                    self.parsed_log[data_id]['prediction'] += prediction_list
                    self.parsed_log[data_id]['double_check'] += double_check_list

    @staticmethod
    def point_adjustment(results, labels, thres_percentage: float = 0.0):
        """
        Adjust anomaly detection results based on the ground-truth labels.
        """
        adjusted_results = results.copy()
        in_anomaly = False
        start_idx = 0

        for i in range(len(labels)):
            if labels[i] == 1 and not in_anomaly:
                in_anomaly = True
                start_idx = i
            elif labels[i] == 0 and in_anomaly:
                in_anomaly = False
                thres = (i - start_idx) * thres_percentage
                if np.sum(results[start_idx:i]) > thres:    # threshold
                    adjusted_results[start_idx:i] = 1

        # Handle the case where the last segment is an anomaly
        if in_anomaly and np.any(results[start_idx:] == 1):
            adjusted_results[start_idx:] = 1
        return adjusted_results

    @staticmethod
    def get_metrics(TP, FP, TN, FN):
        accuracy = (TP + TN) / (TP + FP + TN + FN)
        precision = TP / (TP + FP) if TP + FP != 0 else 0
        recall = TP / (TP + FN) if TP + FN != 0 else 0
        F1_score = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
        return accuracy, precision, recall, F1_score

    @staticmethod
    def get_tpr_fpr(TP, FP, TN, FN):
        TPR = TP / (TP + FN) if TP + FN != 0 else 0
        FPR = FP / (FP + TN) if FP + TN != 0 else 0
        return TPR, FPR

    def calculate_TP_FP_TN_FN(self, confidence_thres=9, thres_percentage:float=0.0, data_id_list:list=[], show_results:bool=False):
        if data_id_list == []:
            data_id_list = self.parsed_log.keys()
        count_log = {}
        
        # print(f"\nDataset: {self.dataset_name}, Confidence Threshold: {confidence_thres}")
        for data_id in data_id_list:
            count_log[data_id] = {
                "pred": {
                    "TP": 0,
                    "FP": 0,
                    "TN": 0,
                    "FN": 0,
                },
                "pred_adjust": {
                    "TP": 0,
                    "FP": 0,
                    "TN": 0,
                    "FN": 0,
                },
                "double_check": {
                    "TP": 0,
                    "FP": 0,
                    "TN": 0,
                    "FN": 0,
                },
                "double_check_adjust": {
                    "TP": 0,
                    "FP": 0,
                    "TN": 0,
                    "FN": 0,
                }
            }
            data_id_log = self.parsed_log[data_id]
            raw_data = data_id_log['raw_data']  # raw data
            raw_label = data_id_log['label']    # [0,1] label array
            pred_list = data_id_log['prediction']
            double_check_list = data_id_log['double_check']

            # vote array
            pred_vote_array = np.zeros(raw_data.shape)
            # confidence array
            pred_confidence_array = np.zeros(raw_data.shape)
            # pred
            for pred_item in pred_list:
                ch = pred_item['channel']
                start = pred_item['start']
                end = pred_item['end']
                confidence = pred_item['confidence']
                type = pred_item['type']
                # vote 
                pred_vote_array[start:end, ch] += 1
                # confidence
                pred_confidence_array[start:end, ch] += confidence
            
            # vote array
            double_check_vote_array = np.zeros(raw_data.shape)
            # confidence array
            double_check_confidence_array = np.zeros(raw_data.shape)
            # double check
            for doubel_check_item in double_check_list:
                ch = doubel_check_item['channel']
                start = doubel_check_item['start']
                end = doubel_check_item['end']
                confidence = doubel_check_item['confidence']
                type = doubel_check_item['type']
                # vote 
                double_check_vote_array[start:end, ch] += 1
                # confidence
                double_check_confidence_array[start:end, ch] += confidence
        
            # threshold
            pred_vote_array = (pred_confidence_array >= confidence_thres).astype(int)
            double_check_vote_array = (double_check_confidence_array >= confidence_thres).astype(int)

            # calculate F1 score for each data_id
            for ch in range(raw_data.shape[1]):
                TP = np.sum((pred_vote_array[:, ch] == 1) & (raw_label[:, ch] == 1))
                FP = np.sum((pred_vote_array[:, ch] == 1) & (raw_label[:, ch] == 0))
                TN = np.sum((pred_vote_array[:, ch] == 0) & (raw_label[:, ch] == 0))
                FN = np.sum((pred_vote_array[:, ch] == 0) & (raw_label[:, ch] == 1))
                count_log[data_id]['pred']['TP'] += TP
                count_log[data_id]['pred']['FP'] += FP
                count_log[data_id]['pred']['TN'] += TN
                count_log[data_id]['pred']['FN'] += FN
                acc, pre, rec, f1 = self.get_metrics(TP, FP, TN, FN)
                # print(f"data_id: {data_id}, channel: {ch}")
                # print(f"\tTP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
                # print(f"\tPrediction >> Acc: {acc:.3f}, Pre: {pre:.3f}, Rec: {rec:.3f}, F1: {f1:.3f}")
                tabel_format = [
                    ["Type", "TP", "FP", "TN", "FN", "Acc", "Pre", "Rec", "F1"],
                    ["Pred", TP, FP, TN, FN, f"{acc:.3f}", f"{pre:.3f}", f"{rec:.3f}", f"{f1:.3f}"],
                ]
                # point-adjustment
                point_adjusted_pred_array = self.point_adjustment(pred_vote_array[:, ch], raw_label[:, ch], thres_percentage)
                TP = np.sum((point_adjusted_pred_array == 1) & (raw_label[:, ch] == 1))
                FP = np.sum((point_adjusted_pred_array == 1) & (raw_label[:, ch] == 0))
                TN = np.sum((point_adjusted_pred_array == 0) & (raw_label[:, ch] == 0))
                FN = np.sum((point_adjusted_pred_array == 0) & (raw_label[:, ch] == 1))
                count_log[data_id]['pred_adjust']['TP'] += TP
                count_log[data_id]['pred_adjust']['FP'] += FP
                count_log[data_id]['pred_adjust']['TN'] += TN
                count_log[data_id]['pred_adjust']['FN'] += FN
                acc, pre, rec, f1 = self.get_metrics(TP, FP, TN, FN)
                tabel_format.append(["Pred(adjust)", TP, FP, TN, FN, f"{acc:.3f}", f"{pre:.3f}", f"{rec:.3f}", f"{f1:.3f}"])

                # double check
                TP = np.sum((double_check_vote_array[:, ch] == 1) & (raw_label[:, ch] == 1))
                FP = np.sum((double_check_vote_array[:, ch] == 1) & (raw_label[:, ch] == 0))
                TN = np.sum((double_check_vote_array[:, ch] == 0) & (raw_label[:, ch] == 0))
                FN = np.sum((double_check_vote_array[:, ch] == 0) & (raw_label[:, ch] == 1))
                count_log[data_id]['double_check']['TP'] += TP
                count_log[data_id]['double_check']['FP'] += FP
                count_log[data_id]['double_check']['TN'] += TN
                count_log[data_id]['double_check']['FN'] += FN
                acc, pre, rec, f1 = self.get_metrics(TP, FP, TN, FN)
                # print(f"data_id: {data_id}, channel: {ch}")
                # print(f"\tTP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
                # print(f"\tDouble_check >> Acc: {acc:.3f}, Pre: {pre:.3f}, Rec: {rec:.3f}, F1: {f1:.3f}")
                tabel_format.append(["DCheck", TP, FP, TN, FN, f"{acc:.3f}", f"{pre:.3f}", f"{rec:.3f}", f"{f1:.3f}"])

                # point-adjustment
                point_adjusted_double_check_array = self.point_adjustment(double_check_vote_array[:, ch], raw_label[:, ch], thres_percentage)
                TP = np.sum((point_adjusted_double_check_array == 1) & (raw_label[:, ch] == 1))
                FP = np.sum((point_adjusted_double_check_array == 1) & (raw_label[:, ch] == 0))
                TN = np.sum((point_adjusted_double_check_array == 0) & (raw_label[:, ch] == 0))
                FN = np.sum((point_adjusted_double_check_array == 0) & (raw_label[:, ch] == 1))
                count_log[data_id]['double_check_adjust']['TP'] += TP
                count_log[data_id]['double_check_adjust']['FP'] += FP
                count_log[data_id]['double_check_adjust']['TN'] += TN
                count_log[data_id]['double_check_adjust']['FN'] += FN
                acc, pre, rec, f1 = self.get_metrics(TP, FP, TN, FN)
                tabel_format.append(["DCheck(adjust)", TP, FP, TN, FN, f"{acc:.3f}", f"{pre:.3f}", f"{rec:.3f}", f"{f1:.3f}"])
                if show_results:
                    print(f"\ndata_id: {data_id}, channel: {ch}")
                    print(tabulate(tabel_format, headers='firstrow', tablefmt='fancy_grid'))

        # calculate F1 score for all data_id
        TP = sum([count_log[data_id]['pred']['TP'] for data_id in count_log])
        FP = sum([count_log[data_id]['pred']['FP'] for data_id in count_log])
        TN = sum([count_log[data_id]['pred']['TN'] for data_id in count_log])
        FN = sum([count_log[data_id]['pred']['FN'] for data_id in count_log])
        acc, pre, rec, f1 = self.get_metrics(TP, FP, TN, FN)
        # print(f"All data_id: ")
        # print(f"\tPrediction >> Acc: {acc:.3f}, Pre: {pre:.3f}, Rec: {rec:.3f}, F1: {f1:.3f}")
        tabel_format = [
            ["Type", "TP", "FP", "TN", "FN", "Acc", "Pre", "Rec", "F1"],
            ["Pred", TP, FP, TN, FN, f"{acc:.3f}", f"{pre:.3f}", f"{rec:.3f}", f"{f1:.3f}"],
        ]
        Pred_item = {
            "TP": TP,
            "FP": FP,
            "TN": TN,
            "FN": FN,
        }
        # adjust
        TP = sum([count_log[data_id]['pred_adjust']['TP'] for data_id in count_log])
        FP = sum([count_log[data_id]['pred_adjust']['FP'] for data_id in count_log])
        TN = sum([count_log[data_id]['pred_adjust']['TN'] for data_id in count_log])
        FN = sum([count_log[data_id]['pred_adjust']['FN'] for data_id in count_log])
        acc, pre, rec, f1 = self.get_metrics(TP, FP, TN, FN)
        tabel_format.append(["Pred(adjust)", TP, FP, TN, FN, f"{acc:.3f}", f"{pre:.3f}", f"{rec:.3f}", f"{f1:.3f}"])
        Pred_adjust_item = {
            "TP": TP,
            "FP": FP,
            "TN": TN,
            "FN": FN,
        }
        # double_check
        TP = sum([count_log[data_id]['double_check']['TP'] for data_id in count_log])
        FP = sum([count_log[data_id]['double_check']['FP'] for data_id in count_log])
        TN = sum([count_log[data_id]['double_check']['TN'] for data_id in count_log])
        FN = sum([count_log[data_id]['double_check']['FN'] for data_id in count_log])
        acc, pre, rec, f1 = self.get_metrics(TP, FP, TN, FN)
        # print(f"All data_id: ")
        # print(f"Double_check >> Acc: {acc:.3f}, Pre: {pre:.3f}, Rec: {rec:.3f}, F1: {f1:.3f}")
        tabel_format.append(["DCheck", TP, FP, TN, FN, f"{acc:.3f}", f"{pre:.3f}", f"{rec:.3f}", f"{f1:.3f}"])
        DCheck_item = {
            "TP": TP,
            "FP": FP,
            "TN": TN,
            "FN": FN,
        }
        # adjust
        TP = sum([count_log[data_id]['double_check_adjust']['TP'] for data_id in count_log])
        FP = sum([count_log[data_id]['double_check_adjust']['FP'] for data_id in count_log])
        TN = sum([count_log[data_id]['double_check_adjust']['TN'] for data_id in count_log])
        FN = sum([count_log[data_id]['double_check_adjust']['FN'] for data_id in count_log])
        acc, pre, rec, f1 = self.get_metrics(TP, FP, TN, FN)
        tabel_format.append(["DCheck(adjust)", TP, FP, TN, FN, f"{acc:.3f}", f"{pre:.3f}", f"{rec:.3f}", f"{f1:.3f}"])
        DCheck_adjust_item = {
            "TP": TP,
            "FP": FP,
            "TN": TN,
            "FN": FN,
        }
        # output
        output_metrics = {
            "Pred": Pred_item,
            "Pred_adjust": Pred_adjust_item,
            "DCheck": DCheck_item,
            "DCheck_adjust": DCheck_adjust_item,
        }
        if show_results:
            print(f"\nAll data_id: ")
            print(tabulate(tabel_format, headers='firstrow', tablefmt='fancy_grid'))
        return output_metrics
    
    def calculate_roc_pr_auc(self,data_id_list:list=[]):
        # self.parsed_log:
        #   - data_id
        #      + raw_data: [global_index, channel]
        #      + label: [global_index, channel]
        #      + prediction: [{'channel': ch, 'start': int, 'end': int, 'confidence': int, 'type': str}, ...]
        #      + double_check: ['channel': ch, 'start': int, 'end': int, 'confidence': int, 'type': str}, ...]
        if data_id_list == []:
            data_id_list = self.parsed_log.keys()
        # auc
        TPR_FPR_map = {
            'Pred': [],
            'Pred_adjust': [],
            'DCheck': [],
            'DCheck_adjust': [],
        }
        PR_map = {
            'Pred': [],
            'Pred_adjust': [],
            'DCheck': [],
            'DCheck_adjust': [],
        }
        AUC_PR_map = {
            'Pred': 0,
            'Pred_adjust': 0,
            'DCheck': 0,
            'DCheck_adjust': 0,
        }
        AUC_ROC_map = {
            'Pred': 0,
            'Pred_adjust': 0,
            'DCheck': 0,
            'DCheck_adjust': 0,
        }
        for conf in range(-2,13):
            res = self.calculate_TP_FP_TN_FN(conf, data_id_list=data_id_list)
            for key in res:
                item = res[key]
                TPR, FPR = self.get_tpr_fpr(item['TP'], item['FP'], item['TN'], item['FN'])
                TPR_FPR_map[key].append((FPR, TPR))
                acc, pre, rec, f1 = self.get_metrics(item['TP'], item['FP'], item['TN'], item['FN'])
                PR_map[key].append((rec, pre))
        # plot ROC curve
        fig, ax = plt.subplots(figsize=(8, 5))
        for key in TPR_FPR_map:
            auc_score = auc([x[0] for x in TPR_FPR_map[key]], [x[1] for x in TPR_FPR_map[key]])
            AUC_ROC_map[key] = auc_score
            ax.plot([x[0] for x in TPR_FPR_map[key]], [x[1] for x in TPR_FPR_map[key]], label=f'{key} (AUC={auc_score:.3f})', marker='x')
        ax.legend()
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(np.arange(0, 1.05, 0.1))
        ax.set_yticks(np.arange(0, 1.05, 0.1))
        ax.grid(True, linestyle='--', color='lightgray', linewidth=0.5)
        ax.set_title(f"{self.dataset_name}-ROC curve")
        fig.savefig(f"./{self.dataset_name}_ROC_curve.png", bbox_inches='tight')
        plt.close()
        # plot PR curve
        fig, ax = plt.subplots(figsize=(7, 5))
        for key in PR_map:
            recall_list = [x[0] for x in PR_map[key]]
            recall_list.append(0)
            precision_list = [x[1] for x in PR_map[key]]
            precision_list.append(1)
            auc_score = auc(recall_list, precision_list)
            AUC_PR_map[key] = auc_score
            ax.plot(recall_list, precision_list, label=f'{key} (AUC={auc_score:.3f})', marker='x')
        ax.legend()
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(np.arange(0, 1.05, 0.1))
        ax.set_yticks(np.arange(0, 1.05, 0.1))
        ax.grid(True, linestyle='--', color='lightgray', linewidth=0.5)
        ax.set_title(f"{self.dataset_name}-PR curve")
        fig.savefig(f"./{self.dataset_name}_PR_curve.png", bbox_inches='tight')
        plt.close()
        return AUC_ROC_map, AUC_PR_map


    def calculate_adjust_PR_curve_auc(self, data_id_list:list=[]):
        if data_id_list == []:
            data_id_list = self.parsed_log.keys()
        # auc
        
        # default_confidence_thres = 9
        fig, ax = plt.subplots(figsize=(13, 7)) 
        auc_score_with_PA_thres = []
        for thres_percentage in np.arange(0, 1.05, 0.2):
            PR_map = {
                # 'Pred': [],
                'Pred_adjust': [],
                # 'DCheck': [],
                # 'DCheck_adjust': [],
            }
            for confidence in range(0, 13):
                res = self.calculate_TP_FP_TN_FN(confidence, thres_percentage, data_id_list)
                for key in PR_map:
                    item = res[key]
                    acc, pre, rec, f1 = self.get_metrics(item['TP'], item['FP'], item['TN'], item['FN'])
                    PR_map[key].append((rec, pre))
            # plot PR curve
            for key in PR_map:
                recall_list = [x[0] for x in PR_map[key]]
                recall_list.append(0)
                precision_list = [x[1] for x in PR_map[key]]
                precision_list.append(1)
                auc_score = auc(recall_list, precision_list)
                auc_score_with_PA_thres.append(auc_score)
                ax.plot(recall_list, precision_list, label=f'Ours (thres={thres_percentage:.2f}, AUC={auc_score:.3f})', marker='x')
                # print(f"{key} >> {PR_map[key]}")
        ax.legend()
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_xlim(0, 1.01)
        ax.set_ylim(0, 1.01)
        ax.set_xticks(np.arange(0, 1.01, 0.1))
        ax.set_yticks(np.arange(0, 1.01, 0.1))
        ax.grid(True, linestyle='--', color='lightgray', linewidth=0.5)
        ax.set_title(f"{self.dataset_name}-PR curve (point-adjustment)")
        fig.savefig(f"./{self.dataset_name}_PR_curve_point_adjustment.png", bbox_inches='tight')
        plt.close()
        return auc_score_with_PA_thres

    def calculate_f1_aucpr_aucroc(self, confidence_thres, point_adjustment_thres, data_id_list:list=[]):
        if data_id_list == []:
            data_id_list = self.parsed_log.keys()
        output_metrics = {}
        res = self.calculate_TP_FP_TN_FN(confidence_thres, point_adjustment_thres, data_id_list)
        auc_roc_map, auc_pr_map = self.calculate_roc_pr_auc(data_id_list)
        table = [
            ['Name', 'P', 'R', 'F1', 'AUCPR', 'AUCROC']
        ]
        # print(data_id_list)
        # for name in res:
        #     pre = res[name]['Pre']
        #     rec = res[name]['Rec']
        #     f1 = res[name]['F1']
        #     aucpr = res[name]['AUC_PR']
        #     aucroc = res[name]['AUC_ROC']
        #     table.append([name, f"{pre:.3f}", f"{rec:.3f}", f"{f1:.3f}", f"{aucpr:.3f}", f"{aucroc:.3f}"])
        # print(tabulate.tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
        for key in res:
            item = res[key]
            acc, pre, rec, f1 = self.get_metrics(item['TP'], item['FP'], item['TN'], item['FN'])
            auc_roc = auc_roc_map[key]
            auc_pr = auc_pr_map[key]
            output_metrics[key] = {
                'Acc': acc,
                'Pre': pre,
                'Rec': rec,
                'F1': f1,
                'AUC_ROC': auc_roc,
                'AUC_PR': auc_pr,
            }
        # print(output_metrics)
        return output_metrics

    def calculate_metrics_with_classification(self, label_root:str, confidence_thres, type_id, data_id_list=[]):
        dir_list = os.listdir(label_root)
        label_path = ''
        for dir_name in dir_list:
            print(dir_name)
            if self.dataset_name in dir_name:
                label_path = os.path.join(label_root, dir_name)
                break
        if label_path == '':
            raise ValueError(f"Dataset name {self.dataset_name} not found in {label_root}")
        count_log = {}
        for data_id in data_id_list:
            # print(data_id)
            count_log[data_id] = {
                "pred": {"TP": 0, "FP": 0, "TN": 0, "FN": 0,},
                "pred_adjust": {"TP": 0, "FP": 0, "TN": 0, "FN": 0,},
                "double_check": {"TP": 0, "FP": 0, "TN": 0, "FN": 0,},
                "double_check_adjust": {"TP": 0, "FP": 0, "TN": 0, "FN": 0,}
            }
            data_id_log = self.parsed_log[data_id]
            raw_data = data_id_log['raw_data']  # raw data
            # raw_label = data_id_log['label']    # [0,1] label array
            raw_label = np.load(os.path.join(label_path, f"{data_id}_annotation.npy"))
            raw_label = raw_label.reshape(-1,1)
            # print(raw_label.shape)
            # print(raw_data.shape);exit()
            # raw_label = raw_label[:raw_data.shape[0]]
            
            pred_list = data_id_log['prediction']
            double_check_list = data_id_log['double_check']
            # vote array
            pred_vote_array = np.zeros(raw_data.shape)
            # confidence array
            pred_confidence_array = np.zeros(raw_data.shape)
            # pred
            for pred_item in pred_list:
                ch = pred_item['channel']
                start = pred_item['start']
                end = pred_item['end']
                confidence = pred_item['confidence']
                type = pred_item['type']
                # vote 
                pred_vote_array[start:end, ch] += 1
                # confidence
                pred_confidence_array[start:end, ch] += confidence
            # vote array
            double_check_vote_array = np.zeros(raw_data.shape)
            # confidence array
            double_check_confidence_array = np.zeros(raw_data.shape)
            # double check
            for doubel_check_item in double_check_list:
                ch = doubel_check_item['channel']
                start = doubel_check_item['start']
                end = doubel_check_item['end']
                confidence = doubel_check_item['confidence']
                type = doubel_check_item['type']
                # vote 
                double_check_vote_array[start:end, ch] += 1
                # confidence
                double_check_confidence_array[start:end, ch] += confidence
            # threshold
            pred_vote_array = (pred_confidence_array >= confidence_thres).astype(int)
            double_check_vote_array = (double_check_confidence_array >= confidence_thres).astype(int)
            # filter by type
            # if type_id not in np.unique(raw_label):
            #     continue
            type_label = raw_label.copy()
            type_label = type_label.astype(int)
            type_label[type_label == 1] = 2
            type_label = (type_label == type_id).astype(int)
            type_pred = pred_vote_array.copy()
            type_pred[type_label == 0] = 0
            type_dcheck = double_check_vote_array.copy()
            type_dcheck[type_label == 0] = 0
            # print(np.where(type_label == 1))
            # print(np.where(type_pred == 1))
            # print(np.where(type_dcheck == 1))
            TP, FP, TN, FN = 0, 0, 0, 0
            TP = np.sum((type_pred[:,0] == 1) & (type_label[:,0] == 1))
            FP = np.sum((type_pred[:,0] == 1) & (type_label[:,0] == 0))
            TN = np.sum((type_pred[:,0] == 0) & (type_label[:,0] == 0))
            FN = np.sum((type_pred[:,0] == 0) & (type_label[:,0] == 1))
            # print(f"{TP}, {FP}, {TN}, {FN}")
            count_log[data_id]['pred']['TP'] += TP
            count_log[data_id]['pred']['FP'] += FP
            count_log[data_id]['pred']['TN'] += TN
            count_log[data_id]['pred']['FN'] += FN
            acc, pre, rec, f1 = self.get_metrics(TP, FP, TN, FN)
            count_log[data_id]['pred']['Acc'] = acc
            count_log[data_id]['pred']['Pre'] = pre
            count_log[data_id]['pred']['Rec'] = rec
            count_log[data_id]['pred']['F1'] = f1
            # print(acc, pre, rec, f1)

            # point-adjustment
            point_adjusted_pred_array = self.point_adjustment(type_pred[:, 0], type_label[:, 0], 0.0)
            TP = np.sum((point_adjusted_pred_array == 1) & (type_label[:,0] == 1))
            FP = np.sum((point_adjusted_pred_array == 1) & (type_label[:,0] == 0))
            TN = np.sum((point_adjusted_pred_array == 0) & (type_label[:,0] == 0))
            FN = np.sum((point_adjusted_pred_array == 0) & (type_label[:,0] == 1))
            # print(f"{TP}, {FP}, {TN}, {FN}")
            count_log[data_id]['pred_adjust']['TP'] += TP
            count_log[data_id]['pred_adjust']['FP'] += FP
            count_log[data_id]['pred_adjust']['TN'] += TN
            count_log[data_id]['pred_adjust']['FN'] += FN
            # print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}");exit()
            acc, pre, rec, f1 = self.get_metrics(TP, FP, TN, FN)
            count_log[data_id]['pred_adjust']['Acc'] = acc
            count_log[data_id]['pred_adjust']['Pre'] = pre
            count_log[data_id]['pred_adjust']['Rec'] = rec
            count_log[data_id]['pred_adjust']['F1'] = f1

            # double_check
            TP = np.sum((type_dcheck[:, 0] == 1) & (type_label[:,0] == 1))
            FP = np.sum((type_dcheck[:, 0] == 1) & (type_label[:,0] == 0))
            TN = np.sum((type_dcheck[:, 0] == 0) & (type_label[:,0] == 0))
            FN = np.sum((type_dcheck[:, 0] == 0) & (type_label[:,0] == 1))
            # print(f"{TP}, {FP}, {TN}, {FN}")
            count_log[data_id]['double_check']['TP'] += TP
            count_log[data_id]['double_check']['FP'] += FP
            count_log[data_id]['double_check']['TN'] += TN
            count_log[data_id]['double_check']['FN'] += FN
            acc, pre, rec, f1 = self.get_metrics(TP, FP, TN, FN)
            count_log[data_id]['double_check']['Acc'] = acc
            count_log[data_id]['double_check']['Pre'] = pre
            count_log[data_id]['double_check']['Rec'] = rec
            count_log[data_id]['double_check']['F1'] = f1

            # point-adjustment
            point_adjusted_double_check_array = self.point_adjustment(type_dcheck[:, 0], type_label[:, 0], 0.0)
            TP = np.sum((point_adjusted_double_check_array == 1) & (type_label[:,0] == 1))
            FP = np.sum((point_adjusted_double_check_array == 1) & (type_label[:,0] == 0))
            TN = np.sum((point_adjusted_double_check_array == 0) & (type_label[:,0] == 0))
            FN = np.sum((point_adjusted_double_check_array == 0) & (type_label[:,0] == 1))
            print(f"{TP}, {FP}, {TN}, {FN}")
            count_log[data_id]['double_check_adjust']['TP'] += TP
            count_log[data_id]['double_check_adjust']['FP'] += FP
            count_log[data_id]['double_check_adjust']['TN'] += TN
            count_log[data_id]['double_check_adjust']['FN'] += FN
            acc, pre, rec, f1 = self.get_metrics(TP, FP, TN, FN)
            count_log[data_id]['double_check_adjust']['Acc'] = acc
            count_log[data_id]['double_check_adjust']['Pre'] = pre
            count_log[data_id]['double_check_adjust']['Rec'] = rec
            count_log[data_id]['double_check_adjust']['F1'] = f1

        # F1 score for all data_id
        output_metrics = {}
        key_list = ['pred', 'pred_adjust', 'double_check', 'double_check_adjust']
        for data_id in count_log:
            table = [
                ["Name", "Acc", "Pre", "Rec", "F1", "Max_F1", "Min_F1", "F1_std"],
            ]
            for key in key_list:
                output_metrics[key] = {}
                TP = sum([count_log[data_id][key]['TP'] for data_id in count_log])
                FP = sum([count_log[data_id][key]['FP'] for data_id in count_log])
                TN = sum([count_log[data_id][key]['TN'] for data_id in count_log])
                FN = sum([count_log[data_id][key]['FN'] for data_id in count_log])
                print(f"{TP}, {FP}, {TN}, {FN}")
                acc, pre, rec, f1 = self.get_metrics(TP, FP, TN, FN)
                output_metrics[key]['Acc'] = acc
                output_metrics[key]['Pre'] = pre
                output_metrics[key]['Rec'] = rec
                output_metrics[key]['F1'] = f1
                # print(f"{key} >> Acc: {acc:.3f}, Pre: {pre:.3f}, Rec: {rec:.3f}, F1: {f1:.3f}")
                output_metrics[key]['Max_F1'] = max([count_log[data_id][key]['F1'] for data_id in count_log])
                output_metrics[key]['Min_F1'] = min([count_log[data_id][key]['F1'] for data_id in count_log])
                output_metrics[key]['F1_std'] = np.std([count_log[data_id][key]['F1'] for data_id in count_log])
                table.append([key, f"{acc:.3f}", f"{pre:.3f}", f"{rec:.3f}", f"{f1:.3f}", f"{output_metrics[key]['Max_F1']:.3f}", f"{output_metrics[key]['Min_F1']:.3f}", f"{output_metrics[key]['F1_std']:.3f}"])
            # print(data_id)
            print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
        return output_metrics
            
if __name__ == '__main__':
    # check_shape('MSL')
    AutoRegister()
    # dataset = RawDataset('UCR', sample_rate=1, normalization_enable=True)
    # dataset.convert_data('./output/test-1-300', 'test', 1000, 500, ImageConvertor)
