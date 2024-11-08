import torch  # PyTorch
import matplotlib.pyplot as plt  # 数据可视化
from data import Data  # 数据加载和预处理
from model import Model  # 模型定义


from monitor import OperatorMonitor
monitor = OperatorMonitor()


# train_window_list = [10, 12, 15, 20]
# epochs_list = [100, 150, 200, 250, 300, 350, 400, 450, 500]
# hidden_layers_list = [100, 150, 200, 250]

# 训练窗口大小
train_window_list = [12, 15]
# 迭代次数
# epochs_list = [500, 600, 700, 800]
epochs_list = [1, 2, 3, 4]
# 隐藏层
hidden_layers_list = [150, 200, 250]
# 批量大小
batch_size_list = [4, 10]
# 测试集比例
test_split = .2

# 遍历批量大小
for batch_size in batch_size_list:
    # 遍历训练窗口大小
    for train_window in train_window_list:
        # 监控数据创建过程的内存占用和执行时间
        with monitor.monitor("LSTM", "create_data", 0):
            # 创建数据和测试序列
            data = Data()
        # 保存数据创建时的监控数据
        # monitor.save_to_csv("operator_monitoring.csv")
        test_split = int((test_split*len(data)))
        train_idx, train_vals = data[:-train_window]
        test_idx, test_vals = data[-train_window:]
        idx_comb = torch.cat((train_idx, test_idx))

        train_inout_seq = data.create_inout_seq(train_vals, train_window=train_window)
        test_inout_seq = data.create_inout_seq(torch.cat((train_vals, test_vals)), train_window=train_window)

        train_dataloader = torch.utils.data.DataLoader(train_inout_seq, batch_size=batch_size, shuffle=False, drop_last=True)

        # 迭代
        for epochs in epochs_list:
            # 隐藏层
            for hidden_layers in hidden_layers_list:
                # 创建模型并训练
                model_name = 'tw' + str(train_window) + '_e' + str(epochs) + '_b' + \
                             str(batch_size) + '_h' + str(hidden_layers)
                print("MODEL: " + model_name)
                # train model
                model = Model(in_size=1, hid=hidden_layers, out_size=1,
                              batch_size=batch_size, seq_len=train_window, monitor=monitor)
                model.fit(train_dataloader, epochs, model_name)


                #train_pred = model.predict(train_vals.tolist(), train_window)
                eval_pred, eval_loss = model.evaluate(test_inout_seq)
                fig, ax = plt.subplots()
                vals_comb = torch.cat((train_vals, test_vals))


                # Evaluate Passenger Model
                raw_train_pred = data.inverse_transform(torch.as_tensor(eval_pred).reshape(-1, 1))
                raw_vals_comb = data.inverse_transform(torch.as_tensor(vals_comb).reshape(-1, 1))

                ax.plot(idx_comb[-len(eval_pred):], raw_train_pred)
                ax.plot(idx_comb, raw_vals_comb)
                ax.set_title('Predicted v. Actual Passenger Counts\nLoss: {}'.format(eval_loss.mean()))
                plt.axvline(x=idx_comb[-1]-train_window, color='g')
                ax.legend(['Predicted', 'Actual'])
                ax.set_ylabel("Passengers")
                plt.savefig("Visuals/" + model_name + "/Comparison.png", dpi=600)
                plt.close()
                print("Prediction Completed: Check Visuals")

                model.save_model(model_name)

                monitor.save_to_csv("operator_monitoring.csv")