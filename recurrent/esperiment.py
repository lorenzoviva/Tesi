from model import LSTM
from utils import read_data



def get_trained_lstm(training_file):
    # Parameters
    learning_rate = 0.001
    # Text file containing words for training
    training_data = read_data(training_file)
    print("Loaded training data...")


    lstm = LSTM(training_data, 3, 512, learning_rate, verbose=True)

    lstm.train(5000)
    return lstm

lstm2 = get_trained_lstm('piccolo_principe2.txt')

# lstm1 = get_trained_lstm('belling_the_cat.txt')

# attenzione, si cancella la prima lettera
#sed "s/\W / &/g" piccolo_principe.txt | sed "s/ \W/& /g" | sed "s/'/ ' /g" | sed "s/  / /g" > piccolo_principe2.txt