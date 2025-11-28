import os
from Datasets.Dataset import Evaluator
import make_dataset
import tabulate
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# ===== Portable CLI + dirs (ADD) =====
def _portable_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--output-dir', default='./output', help='Root for processed data (e.g., generated windows)')
    parser.add_argument('--log-dir', default='./log', help='Root for evaluation logs')
    parser.add_argument('--subtask', default='', help='Optional subfolder under output/ and log/')
    parser.add_argument('--which', default='lmm', choices=['nr', 'window', 'tick', 'lmm'],
                        help='Which evaluation to run')
    parser.add_argument('--datasets', default='UCR',
                        help='Comma-separated dataset names for the chosen evaluation')
    parser.add_argument('--plot', action='store_true', help='Enable plots and PNG saves')
    # knobs for specific evals
    parser.add_argument('--nr-list', default='0,1,2,3,4',
                        help='Comma-separated normal-reference counts for eval_normal_reference')
    parser.add_argument('--tick-list', default='10,25,50,100,200',
                        help='Comma-separated x-tick values for eval_tick')
    parser.add_argument('--data-ids', default='',
                        help='Comma-separated data IDs for eval_tick (empty = all)')
    return parser

def _portable_dirs(args):
    out_dir = (Path(args.output_dir) / args.subtask).resolve()
    log_dir = (Path(args.log_dir) / args.subtask).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, log_dir
# ===== end portable helpers =====

dataset_info_map = make_dataset.dataset_config

# nr_3_map = {
#     'Pred': {
#         'NASA-MSL': {'Pre': 0.739,'Rec': 0.193,'F1': 0.306,'AUC_PR': 0.299,'AUC_ROC': 0.704,},
#         'NASA-SMAP': {'Pre': 0.602,'Rec': 0.388,'F1': 0.472,'AUC_PR': 0.475,'AUC_ROC': 0.781,},
#         'UCR': {'Pre': 0.827,'Rec': 0.774,'F1': 0.8,'AUC_PR': 0.769,'AUC_ROC': 0.961,},
#     },
#     'Pred_adjust': {
#         'NASA-MSL': {'Pre': 0.936,'Rec': 1,'F1': 0.967,'AUC_PR': 0.299,'AUC_ROC': 0.704,},
#         'NASA-SMAP': {'Pre': 0.602,'Rec': 0.388,'F1': 0.472,'AUC_PR': 0.475,'AUC_ROC': 0.781,},
#         'UCR': {'Pre': 0.827,'Rec': 0.774,'F1': 0.8,'AUC_PR': 0.769,'AUC_ROC': 0.961,},
#     },
#     'DCheck': {
#         'NASA-MSL': {'Pre': 0.739,'Rec': 0.193,'F1': 0.306,'AUC_PR': 0.299,'AUC_ROC': 0.704,},
#         'NASA-SMAP': {'Pre': 0.602,'Rec': 0.388,'F1': 0.472,'AUC_PR': 0.475,'AUC_ROC': 0.781,},
#         'UCR': {'Pre': 0.827,'Rec': 0.774,'F1': 0.8,'AUC_PR': 0.769,'AUC_ROC': 0.961,},
#     },
#     'DCheck_adjust': {
#         'NASA-MSL': {'Pre': 0.739,'Rec': 0.193,'F1': 0.306,'AUC_PR': 0.299,'AUC_ROC': 0.704,},
#         'NASA-SMAP': {'Pre': 0.602,'Rec': 0.388,'F1': 0.472,'AUC_PR': 0.475,'AUC_ROC': 0.781,},
#         'UCR': {'Pre': 0.827,'Rec': 0.774,'F1': 0.8,'AUC_PR': 0.769,'AUC_ROC': 0.961,},
#     },
# }

nr_3_map = {
    'NASA-MSL': {
        'Pred': {'Pre': 0.739, 'Rec': 0.193, 'F1': 0.306, 'AUC_PR': 0.299, 'AUC_ROC': 0.704},
        'Pred_adjust': {'Pre': 0.936, 'Rec': 1, 'F1': 0.967, 'AUC_PR': 0.714, 'AUC_ROC': 0.916},
        'DCheck': {'Pre': 0.619, 'Rec': 0.15, 'F1': 0.241, 'AUC_PR': 0.289, 'AUC_ROC': 0.690},
        'DCheck_adjust': {'Pre': 0.916, 'Rec': 1, 'F1': 0.956, 'AUC_PR': 0.733, 'AUC_ROC': 0.921},
    },
    'NASA-SMAP': {
        'Pred': {'Pre': 0.602, 'Rec': 0.388, 'F1': 0.472, 'AUC_PR': 0.475, 'AUC_ROC': 0.781},
        'Pred_adjust': {'Pre': 0.793, 'Rec': 0.983, 'F1': 0.878, 'AUC_PR': 0.892, 'AUC_ROC': 0.970},
        'DCheck': {'Pre': 0.8, 'Rec': 0.395, 'F1': 0.529, 'AUC_PR': 0.586, 'AUC_ROC': 0.828},
        'DCheck_adjust': {'Pre': 0.909, 'Rec': 0.983, 'F1': 0.945, 'AUC_PR': 0.955, 'AUC_ROC': 0.984},
    },
    'UCR': {
        'Pred': {'Pre': 0.827, 'Rec': 0.774, 'F1': 0.8, 'AUC_PR': 0.769, 'AUC_ROC': 0.961},
        'Pred_adjust': {'Pre': 0.861, 'Rec': 1, 'F1': 0.925, 'AUC_PR': 0.930, 'AUC_ROC': 0.998},
        'DCheck': {'Pre': 0.827, 'Rec': 0.774, 'F1': 0.8, 'AUC_PR': 0.773, 'AUC_ROC': 0.967},
        'DCheck_adjust': {'Pre': 0.861, 'Rec': 1, 'F1': 0.925, 'AUC_PR': 0.930, 'AUC_ROC': 0.998},
    },
}

def eval_normal_reference(out_dir: Path, log_dir: Path, dataset_name_list, nr_list, plot_enable=False,
                          default_confidence=9, default_PAT=0):
    eval_log_root = (log_dir / 'Ablation' / 'few-shot')
    processed_data_root = out_dir  # processed data expected at <output>/<dataset>...

    for dataset_name in dataset_name_list:
        dataset_info = dataset_info_map[dataset_name]
        stride = dataset_info['stride']
        for key in ['Pred', 'Pred_adjust', 'DCheck', 'DCheck_adjust']:
            tabel = [['Name', 'P', 'R', 'F1', 'AUCPR', 'AUCROC']]
            plot_data_map = {'F1': [], 'AUCPR': [], 'AUCROC': []}

            for nr in nr_list:
                log_path = str((eval_log_root / f"Ablation-{dataset_name}-NR-{nr}").resolve())
                if nr == 3:
                    metrics = nr_3_map[dataset_name]
                else:
                    evaluator = Evaluator(dataset_name, stride, str(processed_data_root), log_root=log_path)
                    metrics = evaluator.calculate_f1_aucpr_aucroc(default_confidence, default_PAT, data_id_list=[])

                name = f"{key}-{nr}"
                pre = metrics[key]['Pre']; rec = metrics[key]['Rec']; f1 = metrics[key]['F1']
                aucpr = metrics[key]['AUC_PR']; aucroc = metrics[key]['AUC_ROC']
                plot_data_map['F1'].append(f1)
                plot_data_map['AUCPR'].append(aucpr)
                plot_data_map['AUCROC'].append(aucroc)
                tabel.append([name, f"{pre:.3f}", f"{rec:.3f}", f"{f1:.3f}", f"{aucpr:.3f}", f"{aucroc:.3f}"])

            print(f"\nDataset: {dataset_name}, Method: {key}")
            print(tabulate.tabulate(tabel, headers='firstrow', tablefmt='fancy_grid'))

            if plot_enable:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(nr_list, plot_data_map['F1'], label='F1', marker='x')
                ax.plot(nr_list, plot_data_map['AUCPR'], label='AUC-PR', marker='o')
                ax.plot(nr_list, plot_data_map['AUCROC'], label='AUC-ROC', marker='s')
                ax.set_xticks(nr_list)
                ax.set_xlim([min(nr_list), max(nr_list)])
                ax.set_ylim([0, 1])
                ax.set_xlabel('Numbers of normal references')
                ax.set_ylabel('Score')
                ax.set_title(f"Few-shot ablation study on {dataset_name} ({key})")
                ax.grid(linestyle='--', color='gray', alpha=0.5)
                ax.legend()
                fig.savefig(f"ablation_NR_{dataset_name}_{key}.png", bbox_inches='tight')
                plt.close(fig)

                
def eval_window_size(out_dir: Path, log_dir: Path, dataset_list=None, plot_enable=False,
                     default_confidence=9, default_PAT=0):
    if dataset_list is None:
        dataset_list = ['NASA-SMAP']
    p_map = {'UCR': [1, 2, 3, 4], 'NASA-SMAP': [2, 4, 6]}
    period_list = p_map[dataset_list[0]]
    stride_index = [67, 134, 200, 200]  # original mapping

    eval_log_root = (log_dir / 'Ablation' / 'window_size')
    processed_data_root = (out_dir / 'Ablation')  # original script used output/Ablation

    for dataset_name in dataset_list:
        metrics_map = {
            'Pred': {'Pre': [], 'Rec': [], 'F1': [], 'AUC_PR': [], 'AUC_ROC': []},
            'Pred_adjust': {'Pre': [], 'Rec': [], 'F1': [], 'AUC_PR': [], 'AUC_ROC': []},
            'DCheck': {'Pre': [], 'Rec': [], 'F1': [], 'AUC_PR': [], 'AUC_ROC': []},
            'DCheck_adjust': {'Pre': [], 'Rec': [], 'F1': [], 'AUC_PR': [], 'AUC_ROC': []},
        }
        for period in period_list:
            if period in (3, 6):
                metrics = nr_3_map[dataset_name]
            else:
                name = f"{dataset_name}-{period}T"
                stride = stride_index[period // 2 - 1]
                log_path = str((eval_log_root / f"Ablation-{dataset_name}-window_size-{period}T").resolve())
                evaluator = Evaluator(dataset_name, stride, str(processed_data_root),
                                      log_root=log_path, processed_path_name=name)
                metrics = evaluator.calculate_f1_aucpr_aucroc(default_confidence, default_PAT)
            for key in metrics.keys():
                pre = metrics[key]['Pre']; rec = metrics[key]['Rec']; f1 = metrics[key]['F1']
                aucpr = metrics[key]['AUC_PR']; aucroc = metrics[key]['AUC_ROC']
                metrics_map[key]['Pre'].append(pre)
                metrics_map[key]['Rec'].append(rec)
                metrics_map[key]['F1'].append(f1)
                metrics_map[key]['AUC_PR'].append(aucpr)
                metrics_map[key]['AUC_ROC'].append(aucroc)

        for key in metrics_map.keys():
            tabel = [['Name', 'P', 'R', 'F1', 'AUCPR', 'AUCROC']]
            for i, period in enumerate(period_list):
                tabel.append([
                    f"{key}-{period}T",
                    f"{metrics_map[key]['Pre'][i]:.3f}",
                    f"{metrics_map[key]['Rec'][i]:.3f}",
                    f"{metrics_map[key]['F1'][i]:.3f}",
                    f"{metrics_map[key]['AUC_PR'][i]:.3f}",
                    f"{metrics_map[key]['AUC_ROC'][i]:.3f}",
                ])
            print(f"\nDataset: {dataset_name}, Method: {key}")
            print(tabulate.tabulate(tabel, headers='firstrow', tablefmt='fancy_grid'))

            if plot_enable:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(period_list, metrics_map[key]['F1'], label='F1', marker='x')
                ax.plot(period_list, metrics_map[key]['AUC_PR'], label='AUC-PR', marker='o')
                ax.plot(period_list, metrics_map[key]['AUC_ROC'], label='AUC-ROC', marker='s')
                ax.set_xticks(period_list)
                ax.set_xlim([min(period_list), max(period_list) + 0.1])
                ax.set_ylim([0, 1])
                ax.set_xlabel('Window Size (period)', fontsize=18)
                ax.set_ylabel('Score', fontsize=18)
                ax.tick_params(axis='both', which='major', labelsize=18)
                ax.set_title(f"{dataset_name}", fontsize=18)
                ax.grid(linestyle='--', color='gray', alpha=0.5)
                ax.legend(fontsize=18)
                fig.savefig(f"ablation_window_size_{dataset_name}_{key}.png", bbox_inches='tight')
                plt.close(fig)

def eval_tick(out_dir: Path, log_dir: Path, dataset_list=None, data_id_list=None, tick_list=None,
              plot_enable=False, default_confidence=9, default_PAT=0):
    if dataset_list is None:
        dataset_list = ['NormA']
    if data_id_list is None:
        data_id_list = ['1']  # default preserved
    if tick_list is None:
        tick_list = [10, 25, 50, 100, 200]

    eval_log_root = (log_dir / 'Ablation' / 'ticks')
    processed_data_root = (out_dir / 'Ablation')  # original used output/Ablation

    for dataset_name in dataset_list:
        for data_id in data_id_list:
            metrics_map = {
                'Pred': {'Pre': [], 'Rec': [], 'F1': [], 'AUC_PR': [], 'AUC_ROC': []},
                'Pred_adjust': {'Pre': [], 'Rec': [], 'F1': [], 'AUC_PR': [], 'AUC_ROC': []},
                'DCheck': {'Pre': [], 'Rec': [], 'F1': [], 'AUC_PR': [], 'AUC_ROC': []},
                'DCheck_adjust': {'Pre': [], 'Rec': [], 'F1': [], 'AUC_PR': [], 'AUC_ROC': []},
            }
            for tick_value in tick_list:
                name = f"{dataset_name}-tick-{tick_value}"
                log_path = str((eval_log_root / f"Ablation-{dataset_name}-tick-{tick_value}").resolve())
                evaluator = Evaluator(dataset_name, 200, str(processed_data_root),
                                      log_root=log_path, processed_path_name=name)
                metrics = evaluator.calculate_f1_aucpr_aucroc(default_confidence, default_PAT, data_id_list=[])
                for key in metrics.keys():
                    pre = metrics[key]['Pre']; rec = metrics[key]['Rec']; f1 = metrics[key]['F1']
                    aucpr = metrics[key]['AUC_PR']; aucroc = metrics[key]['AUC_ROC']
                    metrics_map[key]['Pre'].append(pre)
                    metrics_map[key]['Rec'].append(rec)
                    metrics_map[key]['F1'].append(f1)
                    metrics_map[key]['AUC_PR'].append(aucpr)
                    metrics_map[key]['AUC_ROC'].append(aucroc)

            for key in metrics_map.keys():
                tabel = [['Name', 'P', 'R', 'F1', 'AUCPR', 'AUCROC']]
                for i, tick_value in enumerate(tick_list[:len(metrics_map[key]['F1'])]):
                    tabel.append([
                        f"{key}-{tick_value}",
                        f"{metrics_map[key]['Pre'][i]:.3f}",
                        f"{metrics_map[key]['Rec'][i]:.3f}",
                        f"{metrics_map[key]['F1'][i]:.3f}",
                        f"{metrics_map[key]['AUC_PR'][i]:.3f}",
                        f"{metrics_map[key]['AUC_ROC'][i]:.3f}",
                    ])
                print(f"\nDataset: {dataset_name}, Method: {key}")
                print(tabulate.tabulate(tabel, headers='firstrow', tablefmt='fancy_grid'))

                if plot_enable:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(tick_list, metrics_map[key]['F1'], label='F1', marker='x')
                    ax.plot(tick_list, metrics_map[key]['AUC_PR'], label='AUC-PR', marker='o')
                    ax.plot(tick_list, metrics_map[key]['AUC_ROC'], label='AUC-ROC', marker='s')
                    ax.set_xticks(tick_list)
                    ax.set_xlim([min(tick_list), max(tick_list) + 0.1])
                    ax.set_ylim([0, 1])
                    ax.set_xlabel('x-ticks', fontsize=14)
                    ax.set_ylabel('Score', fontsize=14)
                    ax.tick_params(axis='both', which='major', labelsize=14)
                    ax.set_title(f"Ticks ablation study on {dataset_name} ({key})", fontsize=16)
                    ax.grid(linestyle='--', color='gray', alpha=0.5)
                    ax.legend(loc='upper right')
                    fig.savefig(f"ablation_ticks_{dataset_name}_{data_id}_{key}.png", bbox_inches='tight')
                    plt.close(fig)

def plot_image():
    save_dir = '/home/zhuangjiaxin/workspace/TensorTSL/TimeLLM/illustrations'
    title = ''
    x = [1, 2, 3, 4]
    F1 = [0.359, 0.552, 0.925]

def LMM_ablation(out_dir: Path, log_dir: Path, dataset_name='UCR', NR_list=None,
                 LLM_list=None, mode='Pred_adjust'):
    if LLM_list is None:
        LLM_list = ['ollama']
    if NR_list is None:
        NR_list = [0, 1]

    dataset_info = make_dataset.dataset_config[dataset_name]
    stride = dataset_info['stride']
    processed_data_root = str(out_dir.resolve())
    log_root = str(log_dir.resolve())

    for nr in NR_list:
        table = [['Model', 'AUC-PR', 'AUC-ROC', 'F1']]
        for LLM_name in LLM_list:
            log_path = os.path.join(log_root, f"{dataset_name}-{LLM_name}-LMM-NR-{nr}")
            evaluator = Evaluator(dataset_name, stride, processed_data_root, log_root=log_path)
            res = evaluator.calculate_f1_aucpr_aucroc(12, 0, data_id_list=[])
            auc_pr = res[mode]['AUC_PR']; auc_roc = res[mode]['AUC_ROC']; f1 = res[mode]['F1']
            table.append([f"{LLM_name}-{nr}", f"{auc_pr:.3f}", f"{auc_roc:.3f}", f"{f1:.3f}"])
        print(f"\nDataset: {dataset_name}, NR: {nr}")
        print(tabulate.tabulate(table, headers='firstrow', tablefmt='fancy_grid'))




if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Portable evaluation runner')
    _portable_args(parser)
    args = parser.parse_args()

    OUT_DIR, LOG_DIR = _portable_dirs(args)

    dataset_names = [d for d in args.datasets.split(',') if d]
    nr_list = [int(x) for x in args.nr_list.split(',') if x]
    tick_list = [int(x) for x in args.tick_list.split(',') if x]
    data_ids = [d for d in args.data_ids.split(',') if d]

    if args.which == 'nr':
        eval_normal_reference(OUT_DIR, LOG_DIR, dataset_names, nr_list, plot_enable=args.plot)
    elif args.which == 'window':
        eval_window_size(OUT_DIR, LOG_DIR, dataset_list=dataset_names, plot_enable=args.plot)
    elif args.which == 'tick':
        eval_tick(OUT_DIR, LOG_DIR, dataset_list=dataset_names,
                  data_id_list=(data_ids or None), tick_list=tick_list,
                  plot_enable=args.plot)
    elif args.which == 'lmm':
        # use first dataset in list for LMM ablation (keeps original behavior)
        ds = dataset_names[0] if dataset_names else 'UCR'
        LMM_ablation(OUT_DIR, LOG_DIR, dataset_name=ds)