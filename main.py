import torch
import matplotlib.pyplot as plt
from data import Data
from model import Model


# train_window_list = [10, 12, 15, 20]
# epochs_list = [100, 150, 200, 250, 300, 350, 400, 450, 500]
# hidden_layers_list = [100, 150, 200, 250]

train_window_list = [12, 15]
epochs_list = [300, 400, 500, 600, 700, 800]
hidden_layers_list = [150, 200, 250]


for train_window in train_window_list:
    data = Data(test_data_size=train_window)
    train_data, test_data = data.get_data()
    train_idx, train_vals = train_data
    test_idx, test_vals = test_data
    train_vals = torch.as_tensor(train_vals.squeeze(), dtype=torch.float32)
    test_vals = torch.as_tensor(test_vals.squeeze(), dtype=torch.float32)
    train_idx, test_idx = train_idx.tolist(), test_idx.tolist()
    idx_comb = train_idx + test_idx

    train_inout_seq = data.create_inout_seq(train_vals)
    test_inout_seq = data.create_inout_seq(torch.cat((train_vals, test_vals)))

    for epochs in epochs_list:
        for hidden_layers in hidden_layers_list:
            model_name = 'tw' + str(train_window) + '_e' + str(epochs) + '_h' + str(hidden_layers)
            print("MODEL: " + model_name)
            # train model
            model = Model(in_size=1, hid=hidden_layers, out_size=1)
            model.fit(train_inout_seq, epochs, model_name)


            #train_pred = model.predict(train_vals.tolist(), train_window)
            eval_pred, eval_loss = model.evaluate(test_inout_seq)
            fig, ax = plt.subplots()
            vals_comb = torch.cat((train_vals, test_vals))


            # Evaluate Passenger Model
            raw_train_pred = data.inverse_transform(torch.as_tensor(eval_pred).reshape(-1, 1))
            raw_vals_comb = data.inverse_transform(torch.as_tensor(vals_comb).reshape(-1, 1))

            ax.plot(idx_comb[-len(eval_pred):], raw_train_pred)
            ax.plot(idx_comb, raw_vals_comb)
            ax.set_title('Predicted v. Actual Passenger Counts\nLoss: {}'.format(eval_loss))
            plt.axvline(x=idx_comb[-1]-train_window, color='g')
            ax.legend(['Predicted', 'Actual'])
            ax.set_ylabel("Passengers")
            plt.savefig("Visuals/" + model_name + "/Comparison.png")
            plt.close()
            print("Prediction Completed: Check Visuals")

            model.save_model(model_name)