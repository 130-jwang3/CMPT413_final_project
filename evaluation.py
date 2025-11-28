import os
from pathlib import Path             # ADD
import argparse                      # ADD
from Datasets.Dataset import EvalDataLoader, Evaluator
import numpy as np
import make_dataset
import tabulate
import yaml
import matplotlib.pyplot as plt

# ---- Portable base (no chdir, no hard-coded C:\) ----
# image rendering config (kept as-is)
image_config = {
    'width': 2000,
    'height': 480,
    'x_ticks': 50,
    'dpi': 100,
}

# Only keep default data_id lists (drop per-dataset log_root here)
DATASET_DEFAULT_IDS = {
    "UCR": ['135','136','137','138'],
    "NASA-MSL": ['3', '9', '10', '11', '15', '23', '24'],
    "NormA": ['1', '4', '7', '13'],
    "NASA-SMAP": ['2', '24', '27', '37', '45'],
    "synthetic_datasets": ['ecg-frequency-0', 'ecg-frequency-1', 'ecg-frequency-2', 'square-frequency-0'],
    "Dodgers": ['data'],
    "ECG": ['CS-MBA-ECG803-data', 'CS-MBA-ECG806-data', 'CS-MBA-ECG820-data',
            'RW-MBA-ECG14046-data-12', 'RW-MBA-ECG14046-data-44', 'RW-MBA-ECG803-data',
            'WN-MBA-ECG14046-data-12', 'WN-MBA-ECG14046-data-5', 'WN-MBA-ECG803-data'],
    "MSD-1": ['machine-1-1-10','machine-1-1-11','machine-1-1-12','machine-1-1-13','machine-1-1-14',
              'machine-1-1-15','machine-1-1-23','machine-1-1-25','machine-1-1-26','machine-1-1-28',
              'machine-1-1-32','machine-1-1-33','machine-1-1-5','machine-1-1-6','machine-1-1-8','machine-1-1-9'],
}

dataset_info_map = make_dataset.dataset_config


# --------- CLI helpers ---------
def _portable_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--output-dir', default='./output', help='Processed data root')
    parser.add_argument('--log-dir', default='./log', help='Evaluation logs root')
    parser.add_argument('--subtask', default='', help='Optional subfolder under output/ and log/')
    parser.add_argument('--dataset', default='UCR', help='Dataset name')
    parser.add_argument('--data-ids', default='', help='Comma-separated list; empty = use defaults for dataset')
    parser.add_argument('--task', default='each_id',
                        choices=['old_eval','each_id','plot_auc_pat','eval_metrics','fix_log','eval_cls'],
                        help='Which function to run')
    parser.add_argument('--stride', type=int, default=-1, help='Override stride (if > 0)')
    parser.add_argument('--key-name', default='Pred_adjust', help='Metric key to highlight (e.g., Pred_adjust)')
    parser.add_argument('--confidence', type=int, default=9, help='Default confidence threshold')
    parser.add_argument('--pat', type=int, default=0, help='Default point-adjust threshold (PAT)')
    parser.add_argument('--baseline-yaml', default='', help='Path to baseline AUC-PR YAML for plot_auc_pat')
    parser.add_argument('--class-root', default='', help='Classification results root for eval_cls')
    parser.add_argument('--original-log', default='', help='Original YAML log path for fix_log')
    parser.add_argument('--save-log', default='', help='Save path for fixed YAML log in fix_log')
    parser.add_argument('--plot', action='store_true', help='Enable plotting where supported')
    return parser

def _portable_dirs(args):
    out_dir = (Path(args.output_dir) / args.subtask).resolve()
    log_dir = (Path(args.log_dir) / args.subtask).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, log_dir

def _resolve_ids(dataset_name: str, override: str):
    if override:
        return [x for x in override.split(',') if x]
    return DATASET_DEFAULT_IDS.get(dataset_name, [])

def old_eval(out_dir: Path, log_dir: Path, dataset_name='MSD-1', point_adjust_enable=True,
             plot_enable=True, channel_shared=False):
    dataset_info = dataset_info_map[dataset_name]
    window_size = dataset_info['window']
    stride = dataset_info['stride']
    log_path = str(log_dir)
    eval_loader = EvalDataLoader(dataset_name, str(out_dir), log_root=log_path)
    eval_loader.set_plot_config(**image_config)
    eval_loader.eval(window_size, stride, vote_thres=2,
                     point_adjust_enable=point_adjust_enable,
                     plot_enable=plot_enable,
                     channel_shared=channel_shared)


def evaluate_each_data_id(out_dir: Path, log_dir: Path, dataset_name='UCR',
                          data_ids=None, key_name='Pred_adjust',
                          stride_override: int = -1,
                          default_confidence_thres: int = 9, default_PAT: int = 0):
    if data_ids is None:
        data_ids = _resolve_ids(dataset_name, '')
    dataset_info = dataset_info_map[dataset_name]
    stride = stride_override if stride_override and stride_override > 0 else dataset_info['stride']

    print(f'SubTask: {log_dir.name}, {dataset_name}, log: {log_dir}, stride: {stride}')
    print(f'Processed data root: {out_dir}')

    evaluator = Evaluator(dataset_name, stride, str(out_dir), log_root=str(log_dir))

    pre_list, rec_list, f1_list, auc_pr_list, auc_roc_list = [], [], [], [], []
    for data_id in data_ids:
        metrics = evaluator.calculate_f1_aucpr_aucroc(default_confidence_thres, default_PAT, data_id_list=[data_id])
        table = [['Name', 'P', 'R', 'F1', 'AUCPR', 'AUCROC']]
        for name in metrics:
            pre = metrics[name]['Pre']; rec = metrics[name]['Rec']; f1 = metrics[name]['F1']
            aucpr = metrics[name]['AUC_PR']; aucroc = metrics[name]['AUC_ROC']
            table.append([name, f"{pre:.3f}", f"{rec:.3f}", f"{f1:.3f}", f"{aucpr:.3f}", f"{aucroc:.3f}"])
        print(tabulate.tabulate(table, headers='firstrow', tablefmt='fancy_grid'))

        pre_list.append(metrics[key_name]['Pre'])
        rec_list.append(metrics[key_name]['Rec'])
        f1_list.append(metrics[key_name]['F1'])
        auc_pr_list.append(metrics[key_name]['AUC_PR'])
        auc_roc_list.append(metrics[key_name]['AUC_ROC'])

    res = evaluator.calculate_f1_aucpr_aucroc(default_confidence_thres, default_PAT, data_id_list=data_ids)
    table = [['Name', 'P', 'R', 'F1', 'AUCPR', 'AUCROC']]
    for name in res:
        pre = res[name]['Pre']; rec = res[name]['Rec']; f1 = res[name]['F1']
        aucpr = res[name]['AUC_PR']; aucroc = res[name]['AUC_ROC']
        table.append([name, f"{pre:.3f}", f"{rec:.3f}", f"{f1:.3f}", f"{aucpr:.3f}", f"{aucroc:.3f}"])
    print(tabulate.tabulate(table, headers='firstrow', tablefmt='fancy_grid'))

    print(f"Method: {key_name}")
    print(f'Precision: {max(pre_list):.3f} / {res[key_name]["Pre"]:.3f} / {np.std(pre_list):.3f}')
    print(f'Recall: {max(rec_list):.3f} / {res[key_name]["Rec"]:.3f} / {np.std(rec_list):.3f}')
    print(f'F1: {max(f1_list):.3f} / {res[key_name]["F1"]:.3f} / {np.std(f1_list):.3f}')
    print(f'AUC_PR: {max(auc_pr_list):.3f} / {res[key_name]["AUC_PR"]:.3f} / {np.std(auc_pr_list):.3f}')
    print(f'AUC_ROC: {max(auc_roc_list):.3f} / {res[key_name]["AUC_ROC"]:.3f} / {np.std(auc_roc_list):.3f}')


# evaluator.calculate_roc_pr_auc(config[dataset_name]['data_id_list'])

def plot_AUC_PR_PAT(out_dir: Path, log_dir: Path, baseline_yaml_path: str, save_yaml: str = 'point_adjustment_auc_pr.yaml'):
    if not baseline_yaml_path:
        raise ValueError('--baseline-yaml is required for plot_auc_pat')
    baseline_info = yaml.safe_load(open(baseline_yaml_path, 'r', encoding='utf-8'))
    save_results = {}

    for dataset_name in baseline_info:
        ds_name = 'MSD-1' if dataset_name in ('SMD', 'SMD-1') else dataset_name
        dataset_info = dataset_info_map[ds_name]
        stride = dataset_info['stride']
        evaluator = Evaluator(ds_name, stride, str(out_dir), log_root=str(log_dir))
        ours_auc_pr = evaluator.calculate_adjust_PR_curve_auc(DATASET_DEFAULT_IDS.get(ds_name, []))

        tabel = [
            ['Name', 'thres=0.0', 'thres=0.2', 'thres=0.4', 'thres=0.6', 'thres=0.8', 'thres=1.0'],
            ['Ours'] + [f'{x:.3f}' for x in ours_auc_pr]
        ]

        for baseline_model in baseline_info[dataset_name]:
            baseline_auc_pr = list(baseline_info[dataset_name][baseline_model])
            baseline_auc_pr.sort(reverse=True)
            tabel.append([baseline_model] + [f'{x:.3f}' for x in baseline_auc_pr])

        print(f"Dataset: {ds_name}")
        print(tabulate.tabulate(tabel, headers='firstrow', tablefmt='fancy_grid'))

        result_item = {'Ours': [f'{x:.3f}' for x in ours_auc_pr]}
        for baseline_model in baseline_info[dataset_name]:
            vals = list(baseline_info[dataset_name][baseline_model])
            vals.sort(reverse=True)
            result_item[baseline_model] = [f'{x:.3f}' for x in vals]
        save_results[ds_name] = result_item

        x_tick = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.plot(x_tick, ours_auc_pr, label='Ours', marker='o', alpha=0.8, markersize=4)
        for baseline_model in baseline_info[dataset_name]:
            vals = list(baseline_info[dataset_name][baseline_model])
            vals.sort(reverse=True)
            ax.plot(x_tick, vals, label=baseline_model, marker='o', alpha=0.8, markersize=4)
        ax.set_xlim([0, 1.01]); ax.set_ylim([0, 1.01])
        ax.set_xlabel('Point-adjustment threshold', fontsize=14)
        ax.set_ylabel('AUC-PR', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_title(f'AUC-PR-PAT curve of {ds_name}', fontsize=16)
        ax.grid(linestyle='--', color='gray', alpha=0.5)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        fig.savefig(f"point-adjustment-auc-pr-{ds_name}.png", bbox_inches='tight')
        plt.close(fig)

    with open(save_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(save_results, f)

def evaluate_metrics(out_dir: Path, log_dir: Path, dataset_name='synthetic_datasets',
                     confidence=6, pat=0):
    dataset_info = dataset_info_map[dataset_name]
    stride = dataset_info['stride']
    print(f'SubTask: {log_dir.name}, {dataset_name}, log: {log_dir}, stride: {stride}')
    print(f'Processed data root: {out_dir}')
    evaluator = Evaluator(dataset_name, stride, str(out_dir), log_root=str(log_dir))
    res = evaluator.calculate_f1_aucpr_aucroc(confidence, pat, data_id_list=DATASET_DEFAULT_IDS.get(dataset_name, []))
    table = [['Name', 'P', 'R', 'F1', 'AUCPR', 'AUCROC']]
    for name in res:
        pre = res[name]['Pre']; rec = res[name]['Rec']; f1 = res[name]['F1']
        aucpr = res[name]['AUC_PR']; aucroc = res[name]['AUC_ROC']
        table.append([name, f"{pre:.3f}", f"{rec:.3f}", f"{f1:.3f}", f"{aucpr:.3f}", f"{aucroc:.3f}"])
    print(tabulate.tabulate(table, headers='firstrow', tablefmt='fancy_grid'))


def fix_log(out_dir: Path, original_log_path: str, save_log_path: str, dataset_name='synthetic_datasets'):
    if not original_log_path or not save_log_path:
        raise ValueError('--original-log and --save-log are required for fix_log')

    from Datasets.Dataset import ProcessedDataset
    import numpy as np

    original_log = yaml.safe_load(open(original_log_path, 'r', encoding='utf-8'))
    dataset = ProcessedDataset(str(out_dir / dataset_name), mode='test')
    for data_id in original_log:
        for stride_idx in original_log[data_id]:
            for ch in original_log[data_id][stride_idx]:
                labels = dataset.get_label(data_id, int(stride_idx), int(ch))
                labels_index = np.where(labels >= 1)[0].tolist()
                original_log[data_id][stride_idx][ch]['labels'] = str(labels_index)

    with open(save_log_path, 'w', encoding='utf-8') as f:
        yaml.dump(original_log, f)

def evaluation_with_classification(out_dir: Path, log_dir: Path, dataset_name='NASA-SMAP',
                                   class_root: str = '', default_confidence_thres: int = 3):
    if not class_root:
        raise ValueError('--class-root is required for eval_cls')
    dataset_info = dataset_info_map[dataset_name]
    stride = dataset_info['stride']
    print(f'{dataset_name}, log: {log_dir}')
    evaluator = Evaluator(dataset_name, stride, str(out_dir), log_root=str(log_dir))
    for type_id in range(1, 6):
        res = evaluator.calculate_metrics_with_classification(class_root, default_confidence_thres,
                                                              type_id,
                                                              data_id_list=DATASET_DEFAULT_IDS.get(dataset_name, []))
        print(f'Type: {type_id}')
        print(res)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Portable evaluation utilities')
    _portable_args(parser)
    args = parser.parse_args()

    OUT_DIR, LOG_DIR = _portable_dirs(args)
    dataset_name = args.dataset
    data_ids = _resolve_ids(dataset_name, args.data_ids)

    if args.task == 'old_eval':
        old_eval(OUT_DIR, LOG_DIR, dataset_name, point_adjust_enable=True,
                 plot_enable=args.plot, channel_shared=False)

    elif args.task == 'each_id':
        evaluate_each_data_id(OUT_DIR, LOG_DIR,
                              dataset_name=dataset_name,
                              data_ids=data_ids,
                              key_name=args.key_name,
                              stride_override=args.stride,
                              default_confidence_thres=args.confidence,
                              default_PAT=args.pat)

    elif args.task == 'plot_auc_pat':
        plot_AUC_PR_PAT(OUT_DIR, LOG_DIR, baseline_yaml_path=args.baseline_yaml)

    elif args.task == 'eval_metrics':
        evaluate_metrics(OUT_DIR, LOG_DIR, dataset_name=dataset_name,
                         confidence=args.confidence, pat=args.pat)

    elif args.task == 'fix_log':
        fix_log(OUT_DIR, original_log_path=args.original_log,
                save_log_path=args.save_log, dataset_name=dataset_name)

    elif args.task == 'eval_cls':
        evaluation_with_classification(OUT_DIR, LOG_DIR,
                                       dataset_name=dataset_name,
                                       class_root=args.class_root,
                                       default_confidence_thres=args.confidence)
