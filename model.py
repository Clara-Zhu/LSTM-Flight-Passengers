import os
import torch
import torch.nn as nn
import tqdm
import data
import numpy as np
import matplotlib.pyplot as plt


class Model(nn.Module):
    def __init__(self, in_size: int = 1, hid: int = 100, num_layers: int = 1, out_size: int = 1, batch_size: int = 1, seq_len: int = 12):
        super().__init__()
        self.in_size = in_size
        self.hidden_size = hid
        self.num_layers = num_layers
        self.out_size = out_size
        self.batch_size = batch_size
        self.seq_len = seq_len

        '''LSTM Expected Dimensions:
        Input: (batch_size, sequence length, input/feature size)
        Hidden States (for forward): (),(): (num_layers*num_directions, batch_size, hidden_size) 
        '''
        self.lstm = nn.LSTM(input_size=in_size, hidden_size=hid, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(in_features=hid, out_features=out_size)
        # self.hidden_cell = (torch.zeros(1, 1, self.hidden_size),
        #                     torch.zeros(1, 1, self.hidden_size))
        self.h_ = torch.zeros(num_layers, batch_size, hid)
        self.c_ = torch.zeros(num_layers, batch_size, hid)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=.001)

    def forward(self, input_seq):
        lstm_out, (self.h_, self.c_) = self.lstm(input_seq, (self.h_, self.c_))
        predictions = self.linear(lstm_out.reshape(-1, self.hidden_size))
        return predictions[-1]

    def fit(self, train_inout_seq: torch.utils.data.DataLoader, epochs: int = 20, model_name: str = 'Default'):
        losses = torch.zeros(epochs, requires_grad=False)
        y_preds = torch.zeros(len(train_inout_seq), requires_grad=False)
        gnd_truth = torch.zeros(len(train_inout_seq), requires_grad=False)

        for i in tqdm.trange(epochs):
            j = 0

            for batch_idx, (seq, label) in enumerate(train_inout_seq):
                self.optimizer.zero_grad()
                self.h_ = torch.zeros(1, self.batch_size, self.hidden_size)
                self.c_ = torch.zeros(1, self.batch_size, self.hidden_size)

                #seq = torch.as_tensor(seq, dtype=torch.float32)
                #label = torch.as_tensor(label.reshape(-1, ), dtype=torch.float32)
                y_pred = self(seq)
                loss = self.loss_fn(y_pred.squeeze(), label[-1].squeeze())
                loss.backward()
                self.optimizer.step()
                if j == len(train_inout_seq) - 1:
                    losses[i] = loss
                if i == epochs - 1:
                    y_preds[j] = y_pred
                    gnd_truth[j] = label[self.batch_size-1].squeeze()
                j += 1

        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 6), sharex=True)
        ax1.plot(y_preds.tolist())
        ax1.plot(gnd_truth.tolist())
        ax2.plot(losses.tolist())
        ax1.set_title("Final Epoch Prediction")
        ax1.set_ylabel("Normalized Training Prediction")
        ax1.legend(['Prediction', 'Actual'])
        ax2.set_title('Model Loss Over Training')
        ax2.set_ylabel("Model Loss")
        ax2.set_xlabel('Epoch')

        if not os.path.isdir("Visuals/" + model_name):
            os.mkdir("Visuals/" + model_name)
        path_plt = "Visuals/" + model_name + "/Model Training.png"
        plt.savefig(path_plt)
        plt.close()
        print("Training Complete: Check " + path_plt)

    def save_model(self, model_name):
        # model_path = 'Models/' + model_name
        # while os.path.isfile(model_path):
        #     choice = input("Model Exists:\n1: Replace\n2: New Model\n")
        #     if choice == '1':
        #         break
        #     elif choice == '2':
        #         name = input("Enter model name\n")
        #         model_path = 'Models/' + name
        # torch.save(self.state_dict(), model_path)
        model_path = 'Models/' + model_name + '.pt'  # 确保模型文件有扩展名
        if os.path.isfile(model_path):
            print(f"Model {model_name} already exists and will be replaced.")
        else:
            print(f"Model {model_name} does not exist and will be created.")
        torch.save(self.state_dict(), model_path)

    def predict(self, test_inputs, train_window: int = 12, fut_pred: int = 12):
        self.eval()
        for i in range(fut_pred):
            seq = torch.FloatTensor(test_inputs[-train_window:])
            with torch.no_grad():
                self.hidden_cell = (torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size))
                test_inputs.append(self(seq).item())
        return test_inputs

    def evaluate(self, test_inout_seq):
        self.eval()
        self.h_ = torch.zeros(self.num_layers, 1, self.hidden_size)
        self.c_ = torch.zeros(self.num_layers, 1, self.hidden_size)
        preds = np.zeros(len(test_inout_seq))
        losses = np.zeros(len(test_inout_seq))
        j = 0
        for seq, label in test_inout_seq:
            # seq = torch.as_tensor(seq)
            #label = torch.as_tensor(label)
            seq = seq.reshape(1, -1, self.in_size)
            preds[j] = self(seq).detach().numpy()[0]
            pred = torch.as_tensor(preds[j])
            lbl = torch.as_tensor(label[0])
            losses[j] = self.loss_fn(pred.squeeze(), lbl.squeeze()).detach().numpy()
            j += 1
        # labels = [x[1].tolist().pop() for x in test_inout_seq]
        return preds, losses

def load_model(model_path: str):
    state_dict = torch.load(model_path)
    hidden_states = int(len(state_dict['lstm.weight_ih_l0'])/4)
    linear_size = len(state_dict['linear.weight'])
    model = Model(in_size=1, hid=hidden_states, num_layers=1, out_size=linear_size)
    model.load_state_dict(state_dict)
    return model
