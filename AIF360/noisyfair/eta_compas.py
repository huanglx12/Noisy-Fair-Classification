# Load all necessary packages
import sys

sys.path.append("../")
import aif360.datasets.noisy_dataset as noisy
import noisyfair.algorithms as algorithms
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, \
    load_preproc_data_compas
import numpy as np
import random
import pandas as pd
import time


def performance(dataset, protected_name, C, lam, eta):
    zvrg_acc = []
    zvrg_sr = []
    gyf_acc = []
    gyf_sr = []
    undenoised_acc = []
    undenoised_sr = []
    denoised_acc = []
    denoised_sr = []
    # generate a noisy dataset
    dataset_train, dataset_test = dataset.split([0.7], shuffle=True)
    for i in range(len(eta)):
        eta0 = eta[i]
        eta1 = eta[i]
        index, noisyfea = noisy.flipping(dataset_train.feature_names, dataset_train.features, protected_name, eta0,
                                         eta1)
        dataset_noisy = np.copy(dataset_train.features)
        dataset_noisy[:, index] = noisyfea
        print("train:", dataset_train.features.shape, 'test:', dataset_test.features.shape)
        print("index:", index)
        print('train_index_1:', sum(dataset_train.features[:, index]), 'train_label_1', sum(dataset_train.labels))
        print('noisy_index_1:', sum(dataset_noisy[:, index]), 'noisy_label_1', sum(dataset_train.labels))
        print('test_index_1:', sum(dataset_test.features[:, index]), 'test_label_1', sum(dataset_test.labels))
        sys.stdout.flush()

        # zvrg
        thresh = 0.2
        print("##########################################")
        print('zvrg:', thresh)
        sys.stdout.flush()
        zvrg_theta = algorithms.zvrg(dataset_noisy, dataset_train.labels, index, C, thresh)
        acc, sr = testing(dataset_test.features, dataset_test.labels, index, zvrg_theta)
        print("acc:", acc, "sr:", sr)
        sys.stdout.flush()
        zvrg_acc.append(acc)
        zvrg_sr.append(sr)

        # gyf
        thresh = 3
        print("##########################################")
        print('gyf:', thresh)
        sys.stdout.flush()
        gyf_theta = algorithms.gyf(dataset_noisy, dataset_train.labels, index, C, thresh)
        acc, sr = testing(dataset_test.features, dataset_test.labels, index, gyf_theta)
        print("acc:", acc, "sr:", sr)
        sys.stdout.flush()
        gyf_acc.append(acc)
        gyf_sr.append(sr)

        # undenoised
        thresh = 0.8
        print("##########################################")
        print('undenoised:', thresh)
        sys.stdout.flush()
        undenoised_theta = algorithms.undenoised(dataset_noisy, dataset_train.labels, index, C, thresh)
        acc, sr = testing(dataset_test.features, dataset_test.labels, index, undenoised_theta)
        print("acc:", acc, "sr:", sr)
        sys.stdout.flush()
        undenoised_acc.append(acc)
        undenoised_sr.append(sr)

        # denoised fair
        thresh = 0.8
        print("##########################################")
        print('denoised:', thresh)
        sys.stdout.flush()
        denoised_theta = algorithms.denoised(dataset_noisy, dataset_train.labels, index, C, thresh, lam, eta0, eta1)
        acc, sr = testing(dataset_test.features, dataset_test.labels, index, denoised_theta)
        print("acc:", acc, "sr:", sr)
        sys.stdout.flush()
        denoised_acc.append(acc)
        denoised_sr.append(sr)

    return zvrg_acc, zvrg_sr, gyf_acc, gyf_sr, undenoised_acc, undenoised_sr, denoised_acc, denoised_sr


################################################################
# report accuracy and fairness on testing dataset
def testing(features, labels, index, theta):
    N = features.shape[0]
    d = features.shape[1]
    N1 = sum(features[:, index])
    N0 = N - N1
    NTrue = 0
    N0True = 0
    N1True = 0
    X = np.zeros([N, d + 1])
    X[:, 0:d] = features
    X[:, d] = [1 for i in range(N)]
    X = np.delete(X, index, 1)
    for i in range(N):
        predict = 1 / (1 + np.exp(-np.dot(theta, X[i])))
        # seed = random.random()
        # if seed < predict:
        if predict >= 0.5:
            predict = 1
        else:
            predict = 0
        if labels[i] == predict:
            NTrue += 1
        if predict == 1:
            if int(features[i][index]) == 1:
                N1True += 1
            else:
                N0True += 1
    acc = NTrue / N
    sr0 = N0True / N0
    sr1 = N1True / N1
    print('sr0:', sr0, 'sr1:', sr1)
    if (sr0 == 0) & (sr1 == 0):
        sr = 1
    elif (sr0 == 0) or (sr1 == 0):
        sr = 0
    else:
        sr = min(sr0 / sr1, sr1 / sr0)
    return acc, sr


###########################################################################
# output an excel
def arrays_write_to_excel(eta, zvrg_acc, zvrg_sr, gyf_acc, gyf_sr,
                          undenoised_acc, undenoised_sr, denoised_acc, denoised_sr, protected_name, times):
    # compute statistics
    zvrg_acc_avg = [np.average(zvrg_acc[:, i]) for i in range(len(eta))]
    zvrg_acc_std = [np.std(zvrg_acc[:, i]) for i in range(len(eta))]
    zvrg_sr_avg = [np.average(zvrg_sr[:, i]) for i in range(len(eta))]
    zvrg_sr_std = [np.std(zvrg_sr[:, i]) for i in range(len(eta))]
    gyf_acc_avg = [np.average(gyf_acc[:, i]) for i in range(len(eta))]
    gyf_acc_std = [np.std(gyf_acc[:, i]) for i in range(len(eta))]
    gyf_sr_avg = [np.average(gyf_sr[:, i]) for i in range(len(eta))]
    gyf_sr_std = [np.std(gyf_sr[:, i]) for i in range(len(eta))]
    undenoised_acc_avg = [np.average(undenoised_acc[:, i]) for i in range(len(eta))]
    undenoised_acc_std = [np.std(undenoised_acc[:, i]) for i in range(len(eta))]
    undenoised_sr_avg = [np.average(undenoised_sr[:, i]) for i in range(len(eta))]
    undenoised_sr_std = [np.std(undenoised_sr[:, i]) for i in range(len(eta))]
    denoised_acc_avg = [np.average(denoised_acc[:, i]) for i in range(len(eta))]
    denoised_acc_std = [np.std(denoised_acc[:, i]) for i in range(len(eta))]
    denoised_sr_avg = [np.average(denoised_sr[:, i]) for i in range(len(eta))]
    denoised_sr_std = [np.std(denoised_sr[:, i]) for i in range(len(eta))]

    excel_dir = 'Compas_' + str(protected_name) + '_eta' + str(times) + '.xlsx'
    with pd.ExcelWriter(excel_dir, engine='xlsxwriter') as writer:
        # zvrg
        l = []
        for i in range(len(eta)):
            l.append([eta[i], zvrg_acc_avg[i], zvrg_acc_std[i], zvrg_sr_avg[i], zvrg_sr_std[i]])
        zvrg_arr = np.asarray(l)
        zvrg_df = pd.DataFrame(zvrg_arr, columns=['eta', 'acc_avg', 'acc_std', 'sr_avg', 'sr_std'])
        zvrg_df.to_excel(writer, sheet_name='zvrg')

        # gyf
        l = []
        for i in range(len(eta)):
            l.append([eta[i], gyf_acc_avg[i], gyf_acc_std[i], gyf_sr_avg[i], gyf_sr_std[i]])
        gyf_arr = np.asarray(l)
        gyf_df = pd.DataFrame(gyf_arr, columns=['eta', 'acc_avg', 'acc_std', 'sr_avg', 'sr_std'])
        gyf_df.to_excel(writer, sheet_name='gyf')

        # undenoised
        l = []
        for i in range(len(eta)):
            l.append([eta[i], undenoised_acc_avg[i], undenoised_acc_std[i], undenoised_sr_avg[i],
                      undenoised_sr_std[i]])
        undenoised_arr = np.asarray(l)
        undenoised_df = pd.DataFrame(undenoised_arr, columns=['eta', 'acc_avg', 'acc_std', 'sr_avg', 'sr_std'])
        undenoised_df.to_excel(writer, sheet_name='undenoised')

        # denoised
        l = []
        for i in range(len(eta)):
            l.append(
                [eta[i], denoised_acc_avg[i], denoised_acc_std[i], denoised_sr_avg[i], denoised_sr_std[i]])
        denoised_arr = np.asarray(l)
        denoised_df = pd.DataFrame(denoised_arr, columns=['eta', 'acc_avg', 'acc_std', 'sr_avg', 'sr_std'])
        denoised_df.to_excel(writer, sheet_name='denoised')
    return


if __name__ == '__main__':
    start = time.time()

    # input
    protected_name = str(sys.argv[1])
    times = int(sys.argv[2])
    print(protected_name, times)
    sys.stdout.flush()

    # initialization
    dataset = load_preproc_data_compas()
    C = 0
    lam = 0.1

    # learn models and predict
    eta = np.linspace(0.1, 0.4, 7, endpoint=True)
    zvrg_acc = np.zeros([times, len(eta)])
    zvrg_sr = np.zeros([times, len(eta)])
    gyf_acc = np.zeros([times, len(eta)])
    gyf_sr = np.zeros([times, len(eta)])
    undenoised_acc = np.zeros([times, len(eta)])
    undenoised_sr = np.zeros([times, len(eta)])
    denoised_acc = np.zeros([times, len(eta)])
    denoised_sr = np.zeros([times, len(eta)])

    for i in range(times):
        print("#####################################################")
        print("times:", i)
        print("#####################################################")
        sys.stdout.flush()
        zvrg_acc[i], zvrg_sr[i], gyf_acc[i], gyf_sr[i], undenoised_acc[i], undenoised_sr[i], denoised_acc[i], \
        denoised_sr[i] = performance(dataset, protected_name, C, lam, eta)

    # record empirical results
    arrays_write_to_excel(eta, zvrg_acc, zvrg_sr, gyf_acc, gyf_sr,
                          undenoised_acc, undenoised_sr, denoised_acc, denoised_sr, protected_name, times)

    print("Execution time:", time.time() - start)
    sys.stdout.flush()


