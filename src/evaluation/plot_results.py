from matplotlib import pyplot as plt
import re
import pandas as pd
import os
import numpy as np
from matplotlib.ticker import MaxNLocator,FormatStrFormatter




def load_results(file_path):
    # Define metric names and target column headers
    metric_names = ["execution", "time acceleration", "exact match"]
    columns = ["easy", "medium", "hard", "extra", "all", "prompt type", "model"]
    
    # Initialize a dictionary to hold the parsed metrics
    parsed = {m: None for m in metric_names}
    
    # Read file content as a single string
    with open(file_path, 'r') as f:
        content = f.read()
    
    # For each metric, search for a line containing the metric name followed by 5 numbers.
    for m in metric_names:
        # Create a regex pattern that looks for the metric name followed by five numeric values.
        pattern = re.compile(
            rf"{m}\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)",
            re.IGNORECASE
        )
        match = pattern.search(content)
        if match:
            parsed[m] = [float(x) for x in match.groups()] + [file_path.split("/")[-1].split(".")[0]] + [file_path.split("/")[-2].split("_")[-1]]
        else:
            print(f"Warning: {m} not found in file.")
    
    # Create a DataFrame with rows as metrics and columns as [easy, medium, hard, extra, all]
    df = pd.DataFrame(parsed, index=columns).T
    return df

def extract_number(prompt_type):
    # Extract the integer inside parentheses
    match = re.search(r'\((\d+)\)', prompt_type)
    return int(match.group(1)) if match else None

def extract_base(prompt_type):
    # Remove any number in parentheses and trim whitespace
    return re.sub(r'\s*\(\d+\)', '', prompt_type).strip()

def extract_group(prompt_type):
    # Remove numbers in parentheses then take the substring before the plus sign
    cleaned = re.sub(r'\(\d+\)', '', prompt_type)
    group = cleaned.split('+')[0].strip()
    return group

def extract_subtype(prompt_type):
    # Return the substring after the plus sign (if it exists) and trim whitespace
    if '+' in prompt_type:
        return prompt_type.split('+')[1].strip()
    return ""

def plot_execution_results(results):
    # Filter only for rows where the metric is 'execution'
    df_exec = results[results.index.str.lower() == 'execution'].copy()
    # Create new columns for the prompt number, base prompt, group, and subtype
    df_exec['prompt_number'] = df_exec['prompt type'].apply(extract_number)
    df_exec['base_prompt'] = df_exec['prompt type'].apply(extract_base)
    df_exec['group'] = df_exec['prompt type'].apply(extract_group)
    df_exec['subtype'] = df_exec['prompt type'].apply(extract_subtype)
    
    # Get unique groups for subplots
    groups = df_exec['group'].unique()
    n = len(groups)
    
    # Mapping for markers: llama -> triangle, ministral -> inverted triangle
    marker_dict = {'llama': 'x', 'ministral': 'o'}
    # Set up a colormap for prompt subtypes
    cmap = plt.get_cmap('tab10')
    
    fig, axes = plt.subplots(1, n-1, figsize=(5*n, 6), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, grp in zip(axes, groups):
        if grp == "results_Zero-shot":
            continue
        sub_group = df_exec[df_exec['group'] == grp]
        # Get unique subtypes in this group to assign colors
        subtypes = sub_group['subtype'].unique()
        color_map = {st: cmap(i, ) for i, st in enumerate(subtypes)}
        line_dict = {st: l for st, l in zip(subtypes, ['-', '--', '-.', ':'])}
        
        # Group by both subtype and model so that curves are separated
        for (subtype, model_name), group_data in sub_group.groupby(['subtype', 'model']):
            group_data = group_data.sort_values('prompt_number')
            marker = marker_dict.get(model_name.lower(), 'o')
            line = line_dict[subtype]
            ax.plot(group_data['prompt_number'],
                    group_data['all'],
                    marker=marker,
                    color=color_map[subtype],
                    linestyle=line,
                    label=f"{subtype} - {model_name}")
        ax.set_title(grp)
        ax.set_xlabel("k (shot)")
        ax.set_ylabel("Execution accuracy (all)")
        ax.legend()
    
    plt.tight_layout()
    plt.savefig("./evaluations/plots/execution_results.png")
    plt.show()

def plot_all_metrics(results):
    # Plot a grid with one row per metric
    metrics = ['execution', 'exact match', 'time acceleration']
    
    # Filter for the three metrics and add new columns
    df_all = results[results.index.str.lower().isin([m.lower() for m in metrics])].copy()
    df_all['prompt_number'] = df_all['prompt type'].apply(extract_number)
    df_all['base_prompt'] = df_all['prompt type'].apply(extract_base)
    df_all['group'] = df_all['prompt type'].apply(extract_group)
    df_all['subtype'] = df_all['prompt type'].apply(extract_subtype)
    
    # Get global groups, excluding "results_Zero-shot"
    global_groups = [g for g in df_all['group'].unique() if g != "results_Zero-shot"]
    n_cols = len(global_groups)
    n_rows = len(metrics)
    
    marker_dict = {'llama': '^', 'ministral': 'v'}
    available_line_styles = ['-', '--', '-.', ':']
    cmap = plt.get_cmap('tab10')
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), sharey='row')
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)
    
    for i, metric in enumerate(metrics):
        # Filter for the current metric (convert index to lower for comparison)
        df_metric = df_all[df_all.index.str.lower() == metric.lower()].copy()
        for j, grp in enumerate(global_groups):
            ax = axes[i, j]
            sub_group = df_metric[df_metric['group'] == grp]
            # Get unique subtypes in this group to assign colors and line styles
            subtypes = sub_group['subtype'].unique()
            color_map = {st: cmap(k) for k, st in enumerate(subtypes)}
            line_style_map = {st: available_line_styles[k % len(available_line_styles)] for k, st in enumerate(subtypes)}
            
            # Group by both subtype and model so that curves are separated
            for (subtype, model_name), group_data in sub_group.groupby(['subtype', 'model']):
                group_data = group_data.sort_values('prompt_number')
                marker = marker_dict.get(model_name.lower(), 'o')
                linestyle = line_style_map.get(subtype, '-')
                ax.plot(group_data['prompt_number'],
                        group_data['all'],
                        marker=marker,
                        color=color_map[subtype],
                        linestyle=linestyle,
                        label=f"{subtype} - {model_name}")
                ax.grid(True)

            if i == 0:
                ax.set_title(grp)
            ax.set_xlabel("k (shot)")
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if j == 0:
                ax.set_ylabel(f"{metric}")
            ax.legend(fontsize='small')
    
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("./evaluations/plots/all_metrics.png")
    plt.show()

def main():

    if not os.path.exists("./evaluations/plots"):
        os.makedirs("./evaluations/plots")

    # Get all files in ./evaluations/
    files_llama = os.listdir("./evaluations/evaluations_llama/")
    files_ministral = os.listdir("./evaluations/evaluations_ministral/")

    # Load results from each file
    results_llama = [load_results(f"./evaluations/evaluations_llama/{f}") for f in files_llama]
    results_ministral = [load_results(f"./evaluations/evaluations_ministral/{f}") for f in files_ministral]

    # Concatenate all results into a single DataFrame
    results_llama = pd.concat(results_llama)
    results_ministral = pd.concat(results_ministral)

    results = pd.concat([results_llama, results_ministral])

    # Plot results
    plot_execution_results(results)
    plot_all_metrics(results)



main()


    

