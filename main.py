
from utils.utils import *
from train_val.training import *
from train_val.testing import *
from sklearn.metrics import classification_report
import torch
import time
import tracemalloc

import numpy as np





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    torch.cuda.empty_cache()

    models=['Custom']




    std_dev_m=[]
    mean_m=[]
    F1score = []
    time_tot = []
    t = 0
    memory=[]
    for m in models:

        files_list, idx, Tags, emo_ratings, emotions=read_data(m)


        Acc=0
        hamming=0

        for i in range(1):
            train_files, val_files, test_files, train_text, val_text, test_text, train_emo, val_emo, test_emo=get_datafiles(files_list, idx, Tags, emo_ratings)
            t=len(test_files)
            #Acc=train(m, train_files, val_files, train_text, val_text, train_emo, val_emo, Acc, i)
            start_time = time.time()

            tracemalloc.start()
            y_predicted, y_a, accuracy, y_scores= testing(m, test_files, test_text, test_emo, i)
            end_time = time.time()
            final_memory = tracemalloc.get_traced_memory()
            final=(final_memory[1]-final_memory[0])/1024**2
            final=final/t
            time_tot.append((end_time-start_time)/t)
            F1score.append(accuracy)
            memory.append(final)
            tracemalloc.stop()


            concatenated_array = np.concatenate((y_predicted, y_a), axis=1)
            df = pd.DataFrame(concatenated_array)
            path='Values_art_'+str(i)+'.csv'
            df.to_csv(path, index=False)

            print(classification_report(y_a, y_predicted, target_names=emotions, zero_division=0))
        print(F1score)






        









