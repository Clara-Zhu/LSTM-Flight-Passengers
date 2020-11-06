import os
import pandas as pd
import torch
import matplotlib.pyplot as plt

import model
from data import Data

def evaluate_models(files, cols):
    df_evals = pd.DataFrame(columns=cols)
    for model_name in files:
        tw, e, h = model_name.split('_')
        tw = int(''.join([x for x in tw if x.isdigit()]))
        e = int(''.join([x for x in e if x.isdigit()]))
        h = int(''.join([x for x in h if x.isdigit()]))

        data = Data(tw)
        train_data, test_data = data.get_data()
        train_idx, train_vals = train_data
        test_idx, test_vals = test_data

        train_vals = torch.as_tensor(train_vals.squeeze(), dtype=torch.float32)
        test_vals = torch.as_tensor(test_vals.squeeze(), dtype=torch.float32)
        train_idx, test_idx = train_idx.tolist(), test_idx.tolist()
        comb_idx = train_idx + test_idx
        comb_vals = torch.cat((train_vals, test_vals))
        comb_inout_seq = data.create_test_seq(comb_vals)

        mdl = model.load_model(model_name)
        print('Evaluating ' + model_name)
        preds, losses = mdl.evaluate(comb_inout_seq)
        dict = {'Model Name': model_name, 'Training Window': tw, 'Epochs': e, 'Hidden Layers': h, 'Losses': losses}
        df_evals = df_evals.append(dict, ignore_index=True)

    print('\n\nWriting to CSV')
    df_evals.to_csv('Evals/Evals.csv')
    return df_evals

def main():
    cols=['Model Name', 'Training Window', 'Epochs', 'Hidden Layers', 'Losses']
    df_evals = pd.DataFrame(columns=cols)

    os.chdir('Models')
    files = [x for x in os.listdir() if os.path.isfile(x)]
    if os.path.isfile('Evals/Evals.csv'):
        confirm = input('CSV Already exists. Re-run evaluation? [y/n]\n')
        if confirm.lower() == 'y':
            df_evals = evaluate_models(files, cols)
        elif confirm.lower() == 'n':
            df_evals = pd.read_csv('Evals.csv')
            df_evals['Losses'] = df_evals['Losses'].apply(lambda x: x.replace('[',''))
            df_evals['Losses'] = df_evals['Losses'].apply(lambda x: x.replace(']', ''))
            df_evals['Losses'] = df_evals['Losses'].apply(lambda x: x.replace('\n', ''))
            df_evals['Losses'] = df_evals['Losses'].apply(lambda x: pd.to_numeric(x.split(' ')))

    else:
        df_evals = evaluate_models(cols)

    # create violin plot of loss distributions for all models
    fig, ax = plt.subplots()
    fig.set_size_inches([19.2,10.8])
    viol = ax.violinplot(df_evals['Losses'], showmeans=True, showextrema=False)
    ax.set_xticks(range(1,len(df_evals['Model Name'])+1))
    ax.set_xticklabels(df_evals['Model Name'], rotation=60., size='x-small', horizontalalignment='right')
    fig.tight_layout()
    plt.title("All Loss Distributions")
    plt.savefig('Evals/Evals.png', dpi=300)

    # Create violin plot for 20 "best" models
    maxlos = [x.max() for x in df_evals['Losses']]
    df_evals['Max Loss'] = maxlos
    dfsmall = df_evals.nsmallest(n=20, columns='Max Loss')
    fig, ax = plt.subplots()
    fig.set_size_inches([19.2, 10.8])
    viol = ax.violinplot(dfsmall['Losses'], showmeans=True, showextrema=False)
    ax.set_xticks(range(1, len(dfsmall['Model Name']) + 1))
    ax.set_xticklabels(dfsmall['Model Name'], rotation=60., size='x-small', horizontalalignment='right')
    fig.tight_layout()
    plt.title('Loss Distribution of 20 Models with Lowest Maximum Loss')
    plt.savefig('Evals/Evals 20 Best.png')



if __name__ == '__main__':
    main()