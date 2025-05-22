import os
import glob
import json
import matplotlib.pyplot as plt
import numpy as np

def load_experiment_data(experiments_dir):
    """
    Загружает данные экспериментов из JSON-файлов в папке experiments_dir,
    имена которых начинаются с восклицательного знака.
    """
    pattern = os.path.join(experiments_dir, '!*.json')
    json_files = glob.glob(pattern)
    
    if not json_files:
        print(f'Не найдено файлов, начинающихся с "!" в папке {experiments_dir}')
    
    experiments = []
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            # Извлекаем имя эксперимента (имя файла без расширения)
            label = os.path.splitext(os.path.basename(file_path))[0]
            experiments.append({'label': label, 'data': data})
        except Exception as e:
            print(f'Ошибка при загрузке файла {file_path}: {e}')
    return experiments

def moving_average(data, window_size):
    """
    Вычисляет скользящее среднее и стандартное отклонение по данным.
    Возвращает два массива: среднее и std, размер которых совпадает с исходным.
    """
    data = np.array(data)
    ma = []
    std = []
    for i in range(len(data)):
        start = max(0, i - window_size + 1)
        window = data[start:i+1]
        ma.append(window.mean())
        std.append(window.std())
    return np.array(ma), np.array(std)

def plot_experiments(experiments):
    """
    Строит 4 подграфика:
      - Верхняя строка: accuracy (обучающая и тестовая) с общим вертикальным масштабом.
      - Нижняя строка: loss (обучающая и тестовая) с общим вертикальным масштабом.
    Для каждого эксперимента строится линия с подписью (label).
    """
    window_size = 390

    # Создаем фигуру с 2 строками и 2 столбцами, 
    # при этом подграфики в каждой строке будут делить ось Y (sharey='row')
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    
    # Заголовки и подписи осей для каждого подграфика
    
    for ax in axs.flat:
        ax.set_xlabel('# full gradient computations')

    axs[1, 0].set_title('Train Accuracy')
    axs[1, 0].set_ylabel('Accuracy')
    
    axs[1, 1].set_title('Test Accuracy')
    axs[1, 1].set_ylabel('Accuracy')
    
    axs[0, 0].set_title('Train Loss')
    axs[0, 0].set_ylabel('Loss')
    
    axs[0, 1].set_title('Test Loss')
    axs[0, 1].set_ylabel('Loss')


    for ax in axs.flat:
        ax.yaxis.set_tick_params(labelleft=True)

    
    # Для каждого эксперимента строим линии на всех подграфиках
    for exp in experiments:
        label = exp['label']
        data = exp['data']
        
        # Проверяем наличие необходимых ключей
        if 'train' in data and 'test' in data:
            train_index = data['train'].get('index', [])
            train_time = data['train'].get('time', [])
            train_accuracy = data['train'].get('accuracy', [])
            train_loss = data['train'].get('loss', [])

            # filtered_train = [(i, t, a, l) for i, t, a, l in zip(train_index, train_time, train_accuracy, train_loss) if i < 44]
            # train_index, train_time, train_accuracy, train_loss = zip(*filtered_train) if filtered_train else ([], [], [], [])

            test_index = data['test'].get('index', [])
            test_time = data['test'].get('time', [])
            test_accuracy = data['test'].get('accuracy', [])
            test_loss = data['test'].get('loss', [])

            # filtered_test = [(i, t, a, l) for i, t, a, l in zip(test_index, test_time, test_accuracy, test_loss) if i < 44]
            # test_index, test_time, test_accuracy, test_loss = zip(*filtered_test) if filtered_test else ([], [], [], [])

            # test_index = test_time
            
            # Строим линии с маркерами для лучшей видимости
            axs[1, 1].plot(test_index, test_accuracy, linestyle="-", label=label)
            axs[0, 1].semilogy(test_index, test_loss, linestyle="-", label=label)

            # Вычисление скользящего среднего и стандартного отклонения для тестовых данных
            ma_acc, std_acc = moving_average(train_accuracy, window_size)
            ma_loss, std_loss = moving_average(train_loss, window_size)
            
            # Построение тестовых графиков с errorbar
            acc_line = axs[1, 0].plot(train_index, ma_acc, linestyle="-", label=label)[0]
            acc_color = acc_line.get_color()
            lss_line = axs[0, 0].semilogy(train_index, ma_loss, linestyle="-", label=label)[0]
            lss_color = lss_line.get_color()

            # Заливаем область от (ma_acc - std_acc) до (ma_acc + std_acc)
            axs[1, 0].fill_between(train_index, 
                                np.array(ma_acc) - 0.5*np.array(std_acc), 
                                np.array(ma_acc) + 0.5*np.array(std_acc), 
                                color=acc_color,
                                alpha=0.1)  # alpha отвечает за прозрачность
            
            # Для потерь используем semilog, поэтому сначала строим линию, затем заливаем область, а потом устанавливаем логарифмическую ось
            axs[0, 0].fill_between(train_index, 
                                np.array(ma_loss) - 0.5*np.array(std_loss), 
                                np.array(ma_loss) + 0.5*np.array(std_loss), 
                                color=lss_color,
                                alpha=0.1)
            axs[0, 0].set_yscale('log')
        else:
            print(f'В данных эксперимента {label} отсутствуют необходимые ключи.')
    
    for ax in axs.flat:
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('experiments_plot.png')
    # plt.show()

def plotacc(experiments_data, save_path="acc_topk.png"):
    import re

    k_values = [1, 2, 3, 4, 5]
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for idx, k in enumerate(k_values):
        ax = axes[idx]
        for exp in experiments_data:
            full_label = exp['label']
            name = full_label.lstrip("!").split("_")[0]  # убираем ! и всё после первого _
            res = exp['data'].get('test', {})
            if "acc_at_k" in res and "index" in res:
                topk_list = [epoch_acc.get(f"top@{k}", 0) for epoch_acc in res["acc_at_k"]]
                index = res["index"]
                if len(index) == len(topk_list):
                    ax.plot(index, topk_list, label=name)
        ax.set_title(f"Top@{k}")
        ax.set_xlabel("Full Grad Steps")
        ax.set_ylim(0.8, 1.0)
        if idx == 0:
            ax.set_ylabel("Accuracy")
        ax.grid(True)
        ax.legend(fontsize='small')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    # Таблица финальных и максимальных значений в процентах
    print("\nМетод\t\tFinAcc@1\tMaxAcc@1\tFinAcc@3\tMaxAcc@3")
    print("-" * 80)
    for exp in experiments_data:
        full_label = exp['label']
        name = full_label.lstrip("!").split("_")[0]
        res = exp['data'].get('test', {})
        if "acc_at_k" in res and res["acc_at_k"]:
            top1_all = [entry.get("top@1", 0) * 100 for entry in res["acc_at_k"]]
            top3_all = [entry.get("top@3", 0) * 100 for entry in res["acc_at_k"]]
            acc1_final = top1_all[-1]
            acc3_final = top3_all[-1]
            acc1_max = max(top1_all)
            acc3_max = max(top3_all)
            print(f"{name:<12}\t{acc1_final:.2f}%\t\t{acc1_max:.2f}%\t\t{acc3_final:.2f}%\t\t{acc3_max:.2f}%")

from matplotlib.ticker import AutoMinorLocator
from matplotlib.backends.backend_pdf import PdfPages

def plot_test_accuracy_pdf(experiments_data, save_path="test_accuracy.pdf"):
    fig, ax = plt.subplots(figsize=(10, 6))

    for exp in experiments_data:
        full_label = exp['label']
        name = full_label.lstrip("!").split("_")[0]  # Только между ! и _
        res = exp['data'].get('test', {})
        if "accuracy" in res and "index" in res:
            acc = res["accuracy"]
            index = res["index"]
            if len(index) == len(acc):
                ax.plot(index, acc, label=name)

    ax.set_title("Test Accuracy")
    ax.set_xlabel("Full Gradient Steps")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.4, 0.9)
    ax.grid(True, which='major', linestyle='-', linewidth=0.5)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.3)
    ax.legend(fontsize='small')

    # Добавим subticks по оси X и Y
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    # Сохраняем в PDF
    with PdfPages(save_path) as pdf:
        pdf.savefig(fig)
    plt.close()



if __name__ == '__main__':
    experiments_directory = 'experiments'
    experiments_data = load_experiment_data(experiments_directory)    
    if experiments_data:
        plot_experiments(experiments_data)
        # plotacc(experiments_data)
        plot_test_accuracy_pdf(experiments_data)

