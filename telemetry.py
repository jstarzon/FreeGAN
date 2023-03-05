from tqdm import tqdm
from csv import writer
import datetime

#Current date
def date():
    now = datetime.datetime.now()
    return now
#CSV Telemetry
now = date() 
def data_csv(epoch, d_loss, d_acc, g_loss):
    List = [now.strftime("%m/%d/%Y, %H:%M:%S") , epoch, d_loss, d_acc, g_loss]
    with open('epoch_info.csv', 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(List)
        f_object.close()