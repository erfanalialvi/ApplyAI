.. code:: ipython3

    #Importing Libraries
    # system imports
    import glob
    import os
    import nbconvert
    
    # feature extractoring and preprocessing data
    import librosa
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline
    pd.plotting.register_matplotlib_converters()
    from PIL import Image
    import pathlib
    import csv
    from tqdm.auto import tqdm
    from sklearn.utils import resample
    import seaborn as sns
    
    #Keras
    import keras
    import warnings
    warnings.filterwarnings('ignore')
    
    # Preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, scale, StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.preprocessing import MinMaxScaler
    from sklearn import metrics
    from sklearn.model_selection import KFold
    import librosa, librosa.display, os, csv
    import matplotlib
    import pylab
    plt.switch_backend('agg')
    import itertools
    import scipy as sp
    from scipy import signal
    import joblib
    from glob import glob
    import urllib
    
    # internal imports
    from utils import preproces, CoughNet

.. code:: ipython3

    # Kaggle Dataset - Exploration
    fn_dataset = 'data/cough_trial_extended.csv'
    df_dataset = pd.read_csv(fn_dataset)
    
    print('Total number of examples:', len(df_dataset))
    print('Number of positive examples:', len(df_dataset[df_dataset['class'] == 'covid']))
    print('Number of negative examples:', len(df_dataset[df_dataset['class'] == 'not_covid']))
    
    df_dataset


.. parsed-literal::

    Total number of examples: 170
    Number of positive examples: 19
    Number of negative examples: 151
    



.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>file_properties</th>
          <th>class</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0v8MGxNetjg_ 10.000_ 20.000.wav</td>
          <td>not_covid</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1j1duoxdxBg_ 70.000_ 80.000.wav</td>
          <td>not_covid</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1MSYO4wgiag_ 120.000_ 130.000.wav</td>
          <td>not_covid</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1PajbAKd8Kg_ 0.000_ 10.000.wav</td>
          <td>not_covid</td>
        </tr>
        <tr>
          <th>4</th>
          <td>cov1.wav</td>
          <td>covid</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>165</th>
          <td>-bZrDCS8KAg_ 70.000_ 80.000.wav</td>
          <td>not_covid</td>
        </tr>
        <tr>
          <th>166</th>
          <td>-ej81N6Aqo4_ 0.000_ 8.000.wav</td>
          <td>not_covid</td>
        </tr>
        <tr>
          <th>167</th>
          <td>-gvLnl1smfs_ 90.000_ 100.000.wav</td>
          <td>not_covid</td>
        </tr>
        <tr>
          <th>168</th>
          <td>-hu5q-Nn4BM_ 70.000_ 80.000.wav</td>
          <td>not_covid</td>
        </tr>
        <tr>
          <th>169</th>
          <td>-jLQkyDhIxw_ 10.000_ 20.000.wav</td>
          <td>not_covid</td>
        </tr>
      </tbody>
    </table>
    <p>170 rows × 2 columns</p>
    </div>



.. code:: ipython3

    # Kaggle Dataset - Feature Extraction
    df_features_cols = ['filename', 'chroma_stft', 'rmse', 'spectral_centroid', 'spectral_bandwidth', 'rolloff', 'zero_crossing_rate']
    for i in range(1, 21):
        df_features_cols.append(f'mfcc{i}')
    df_features_cols.append('label')
    
    df_features = pd.DataFrame(columns=df_features_cols)
    
    for row_index, row in tqdm(df_dataset.iterrows(), total=len(df_dataset)):
        fn_wav = os.path.join('data/trial_covid/', row['file_properties'])
        feature_row = preproces(fn_wav)
        feature_row['filename'] = row['file_properties']
        feature_row['label'] = row['class']
        df_features = df_features.append(feature_row, ignore_index=True)
    
    df_features.to_csv('data/prepared_data_kaggle.csv', index=False, columns=df_features_cols)
    
    df_features.head()



.. parsed-literal::

      0%|          | 0/170 [00:00<?, ?it/s]




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>filename</th>
          <th>chroma_stft</th>
          <th>rmse</th>
          <th>spectral_centroid</th>
          <th>spectral_bandwidth</th>
          <th>rolloff</th>
          <th>zero_crossing_rate</th>
          <th>mfcc1</th>
          <th>mfcc2</th>
          <th>mfcc3</th>
          <th>...</th>
          <th>mfcc12</th>
          <th>mfcc13</th>
          <th>mfcc14</th>
          <th>mfcc15</th>
          <th>mfcc16</th>
          <th>mfcc17</th>
          <th>mfcc18</th>
          <th>mfcc19</th>
          <th>mfcc20</th>
          <th>label</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0v8MGxNetjg_ 10.000_ 20.000.wav</td>
          <td>0.519951</td>
          <td>0.045853</td>
          <td>1612.895795</td>
          <td>1411.838677</td>
          <td>2907.580566</td>
          <td>0.107019</td>
          <td>-376.876007</td>
          <td>111.017372</td>
          <td>-31.904015</td>
          <td>...</td>
          <td>-7.439712</td>
          <td>-1.034580</td>
          <td>-0.203083</td>
          <td>-3.513495</td>
          <td>-1.745705</td>
          <td>-3.011878</td>
          <td>-2.878482</td>
          <td>-2.106427</td>
          <td>-4.026825</td>
          <td>not_covid</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1j1duoxdxBg_ 70.000_ 80.000.wav</td>
          <td>0.535472</td>
          <td>0.001771</td>
          <td>2892.087076</td>
          <td>2467.408141</td>
          <td>5072.664388</td>
          <td>0.148584</td>
          <td>-519.158447</td>
          <td>60.781284</td>
          <td>-13.722886</td>
          <td>...</td>
          <td>-0.909973</td>
          <td>7.216461</td>
          <td>-1.719629</td>
          <td>3.903021</td>
          <td>3.653039</td>
          <td>3.043882</td>
          <td>2.439957</td>
          <td>2.781968</td>
          <td>2.195162</td>
          <td>not_covid</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1MSYO4wgiag_ 120.000_ 130.000.wav</td>
          <td>0.496666</td>
          <td>0.033657</td>
          <td>3429.061935</td>
          <td>2788.634413</td>
          <td>6886.288452</td>
          <td>0.225315</td>
          <td>-282.297913</td>
          <td>48.581680</td>
          <td>-15.522366</td>
          <td>...</td>
          <td>-6.066336</td>
          <td>-4.167640</td>
          <td>1.017302</td>
          <td>-0.523806</td>
          <td>0.538693</td>
          <td>-8.855953</td>
          <td>-2.927977</td>
          <td>-1.118562</td>
          <td>-5.906228</td>
          <td>not_covid</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1PajbAKd8Kg_ 0.000_ 10.000.wav</td>
          <td>0.407549</td>
          <td>0.013452</td>
          <td>2710.811637</td>
          <td>2664.287550</td>
          <td>5778.474935</td>
          <td>0.142076</td>
          <td>-346.857300</td>
          <td>75.765617</td>
          <td>-7.648194</td>
          <td>...</td>
          <td>5.053118</td>
          <td>-0.291308</td>
          <td>0.987186</td>
          <td>-2.447526</td>
          <td>3.692367</td>
          <td>2.312328</td>
          <td>-2.059656</td>
          <td>-4.772599</td>
          <td>-0.503851</td>
          <td>not_covid</td>
        </tr>
        <tr>
          <th>4</th>
          <td>cov1.wav</td>
          <td>0.412697</td>
          <td>0.059004</td>
          <td>1555.648634</td>
          <td>1418.599932</td>
          <td>2870.737092</td>
          <td>0.133998</td>
          <td>-340.588013</td>
          <td>104.156700</td>
          <td>-32.228443</td>
          <td>...</td>
          <td>-8.247168</td>
          <td>0.940006</td>
          <td>-5.701087</td>
          <td>-6.326630</td>
          <td>-1.080040</td>
          <td>-1.812609</td>
          <td>-2.518986</td>
          <td>-3.684266</td>
          <td>-3.564146</td>
          <td>covid</td>
        </tr>
      </tbody>
    </table>
    <p>5 rows × 28 columns</p>
    </div>



.. code:: ipython3

    #Virufy Dataset - Exploration
    import glob
    df_dataset = pd.DataFrame(columns=['file_properties', 'class'])
    for fn in glob.glob('data/virufy/pos/*.mp3'):
        df_dataset = df_dataset.append({'file_properties': fn, 'class': 'covid'}, ignore_index=True)
    for fn in glob.glob('data/virufy/neg/*.mp3'):
        df_dataset = df_dataset.append({'file_properties': fn, 'class': 'not_covid'}, ignore_index=True)
    
    print('Total number of examples:', len(df_dataset))
    print('Number of positive examples:', len(df_dataset[df_dataset['class'] == 'covid']))
    print('Number of negative examples:', len(df_dataset[df_dataset['class'] == 'not_covid']))
    
    df_dataset


.. parsed-literal::

    Total number of examples: 121
    Number of positive examples: 48
    Number of negative examples: 73
    



.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>file_properties</th>
          <th>class</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>data/virufy/pos\pos-0421-084-cough-m-50-0.mp3</td>
          <td>covid</td>
        </tr>
        <tr>
          <th>1</th>
          <td>data/virufy/pos\pos-0421-084-cough-m-50-1.mp3</td>
          <td>covid</td>
        </tr>
        <tr>
          <th>2</th>
          <td>data/virufy/pos\pos-0421-084-cough-m-50-2.mp3</td>
          <td>covid</td>
        </tr>
        <tr>
          <th>3</th>
          <td>data/virufy/pos\pos-0421-084-cough-m-50-3.mp3</td>
          <td>covid</td>
        </tr>
        <tr>
          <th>4</th>
          <td>data/virufy/pos\pos-0421-084-cough-m-50-4.mp3</td>
          <td>covid</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>116</th>
          <td>data/virufy/neg\neg-0422-097-cough-m-37-8.mp3</td>
          <td>not_covid</td>
        </tr>
        <tr>
          <th>117</th>
          <td>data/virufy/neg\neg-0422-097-cough-m-37-9.mp3</td>
          <td>not_covid</td>
        </tr>
        <tr>
          <th>118</th>
          <td>data/virufy/neg\neg-0422-098-cough-f-24-0.mp3</td>
          <td>not_covid</td>
        </tr>
        <tr>
          <th>119</th>
          <td>data/virufy/neg\neg-0422-098-cough-f-24-1.mp3</td>
          <td>not_covid</td>
        </tr>
        <tr>
          <th>120</th>
          <td>data/virufy/neg\neg-0422-098-cough-f-24-5.mp3</td>
          <td>not_covid</td>
        </tr>
      </tbody>
    </table>
    <p>121 rows × 2 columns</p>
    </div>



.. code:: ipython3

    #Virufy Dataset - Feature Extraction
    #df_features_cols = ['filename', 'chroma_stft', 'rmse', 'spectral_centroid', 'spectral_bandwidth', 'rolloff', 'zero_crossing_rate']
    #for i in range(1, 21):
        #df_features_cols.append(f'mfcc{i}')
    #df_features_cols.append('label')
    
    #df_features = pd.DataFrame(columns=df_features_cols)
    
    #for row_index, row in tqdm(df_dataset.iterrows(), total=len(df_dataset)):
        #fn_wav = os.path.join('data/virufy/pos/*.mp3', row['file_properties'])
        #fn_wav = row['file_properties']
        #feature_row = preproces(fn_wav)
        #feature_row['filename'] = row['file_properties']
        #feature_row['label'] = row['class']
        #df_features = df_features.append(feature_row, ignore_index=True)
    
    #df_features.to_csv('data/prepared_data_virufy.csv', index=False, columns=df_features_cols)
    
    #df_features.head()

.. code:: ipython3

    #Combine Datasets
    df_features_kaggle = pd.read_csv('data/prepared_data_kaggle.csv')
    df_features_virufy = pd.read_csv('data/prepared_data_virufy.csv')
    df_features = pd.concat([df_features_kaggle, df_features_virufy])
    
    df_features.to_csv('data/prepared_data.csv', index=False, columns=df_features_cols)
    
    print('Total number of examples:', len(df_features))
    print('Number of positive examples:', len(df_features[df_features['label'] == 'covid']))
    print('Number of negative examples:', len(df_features[df_features['label'] == 'not_covid']))


.. parsed-literal::

    Total number of examples: 291
    Number of positive examples: 67
    Number of negative examples: 224
    

.. code:: ipython3

    #Balanced Dataset
    df_features = pd.read_csv('data/prepared_data.csv')
    
    # Separate majority and minority classes
    df_majority = df_features[df_features['label'] == 'not_covid']
    df_minority = df_features[df_features['label'] == 'covid']
     
    # Downsample majority class
    df_majority_balanced = resample(df_majority, replace=False, n_samples=len(df_minority), random_state=42)
     
    # Combine minority class with downsampled majority class
    df_balanced = pd.concat([df_majority_balanced, df_minority])
    
    df_balanced.to_csv('data/prepared_data_balanced.csv', index=False)
    
    print('Total number of examples:', len(df_balanced))
    print('Number of positive examples:', len(df_balanced[df_balanced['label'] == 'covid']))
    print('Number of negative examples:', len(df_balanced[df_balanced['label'] == 'not_covid']))


.. parsed-literal::

    Total number of examples: 134
    Number of positive examples: 67
    Number of negative examples: 67
    

.. code:: ipython3

    #Training and Evaluation
    # system imports
    
    from datetime import datetime
    
    import torch
    from torch.utils.tensorboard import SummaryWriter
    
    # internal imports
    from utils import plot_confusion_matrix
    
    # device config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

.. code:: ipython3

    #Hyperparameters
    hparams = {    
        'dataset': 'data/prepared_data_balanced.csv',
        'epochs': 15,
        'batch_size': 16,
        'lr': 1e-3,
        'features': [
            'chroma_stft', 'rmse', 'spectral_centroid', 'spectral_bandwidth', 'rolloff', 'zero_crossing_rate',
            'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 
            'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20'
        ]
    }

.. code:: ipython3

    #Prepare Data
    df_features = pd.read_csv(hparams['dataset'])
    
    X = np.array(df_features[hparams['features']], dtype=np.float32)
    
    encoder = LabelEncoder()
    y = encoder.fit_transform(df_features['label'])
    print('classes:', encoder.classes_)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print('X_train.shape:', X_train.shape)
    print('y_train.shape:', y_train.shape)
    
    # create pytorch dataloader
    torch.manual_seed(42)
    train_dataset = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train).long())
    test_dataset = torch.utils.data.TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test).long())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=hparams['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=hparams['batch_size'], shuffle=False)


.. parsed-literal::

    classes: ['covid' 'not_covid']
    X_train.shape: (107, 26)
    y_train.shape: (107,)
    

.. code:: ipython3

    #Setup Model
    # Design model (input, output size, forward pass)
    class CoughNet(torch.nn.Module):
        def __init__(self, input_size):
            super(CoughNet, self).__init__()
            self.l1 = torch.nn.Linear(input_size, 512)
            self.l2 = torch.nn.Linear(512, 256)
            self.l3 = torch.nn.Linear(256, 128)
            self.l4 = torch.nn.Linear(128, 64)
            self.l5 = torch.nn.Linear(64, 10)
            self.l6 = torch.nn.Linear(10, 2)
    
        def forward(self, x):
            x = torch.relu(self.l1(x))
            x = torch.relu(self.l2(x))
            x = torch.relu(self.l3(x))
            x = torch.relu(self.l4(x))
            x = torch.relu(self.l5(x))
            x = self.l6(x)
            return x
    
    model = CoughNet(len(hparams['features'])).to(device)

.. code:: ipython3

    #Training
    # Construct loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
    criterion = torch.nn.CrossEntropyLoss()
    
    def train(loader_train, model, optimizer, epoch):
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        total = 0
        pbar = tqdm(enumerate(loader_train), total=len(loader_train))
        for batch_ndx, sample in pbar: 
            features, labels = sample[0].to(device), sample[1].to(device) 
    
            # forward pass and loss calculation
            outputs = model(features)
            loss = criterion(outputs, labels)  
            
            # backward pass    
            loss.backward()
            
            # update weights
            optimizer.step()
            optimizer.zero_grad()
            
            # calculate metrics
            running_loss += loss.item()
            predictions = torch.argmax(outputs.data, 1)
            running_correct += (predictions == labels).sum().item()
    
            # print informations
            pbar.set_description(f'[Training Epoch {epoch+1}]') 
            total += labels.shape[0]
            pbar.set_postfix({'loss': running_loss / total, 'train_accuracy': running_correct / total})
            
        # write informations to tensorboard
        writer.add_scalar('Loss/Train', running_loss / total, epoch+1)
        writer.add_scalar('Accuracy/Train', running_correct / total, epoch+1)
    
    def evaluate(loader_test, model, epoch):
        model.eval()
        running_loss = 0.0
        running_correct = 0.0
        total = 0
        with torch.no_grad():
            pbar = tqdm(enumerate(loader_test), total=len(loader_test))
            for batch_ndx, sample in pbar:
                features, labels = sample[0].to(device), sample[1].to(device) 
    
                # forward pass and loss calculation
                outputs = model(features)
                loss = criterion(outputs, labels)  
    
                # calculate metrics
                running_loss += loss.item()
                predictions = torch.argmax(outputs.data, 1)
                running_correct += (predictions == labels).sum().item()
    
                # print informations
                pbar.set_description(f'[Evaluating Epoch {epoch+1}]')
                total += labels.shape[0]
                pbar.set_postfix({'loss': running_loss / total, 'eval_accuracy': running_correct / total})
            
        # write informations to tensorboard
        writer.add_scalar('Loss/Eval', running_loss / total, epoch+1)
        writer.add_scalar('Accuracy/Eval', running_correct / total, epoch+1)
    
    # initialize tensorboard summary writer
    time_stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    writer = SummaryWriter(f'logs/{time_stamp}/')
    
    # add graph to tensorboard
    features = iter(test_loader).next()[0]
    writer.add_graph(model, features)
    
    # training loop
    for epoch in range(hparams['epochs']):
        train(train_loader, model, optimizer, epoch)
        evaluate(test_loader, model, epoch)
    
    # close tensorboard
    writer.close()
    
    # open tensorboard
    # tensorboard --logdir logs


::


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_12580\3309310846.py in <module>
         71 
         72 # add graph to tensorboard
    ---> 73 features = iter(test_loader).next()[0]
         74 writer.add_graph(model, features)
         75 
    

    AttributeError: '_SingleProcessDataLoaderIter' object has no attribute 'next'


.. code:: ipython3

    #Plot confusion matrix
    # internal imports
    from utils import plot_confusion_matrix
    model.eval()
    with torch.no_grad():
        outputs = model(torch.tensor(X_test))
        predictions = torch.argmax(outputs.data, 1)
    
    plot_confusion_matrix(y_test, predictions, encoder.classes_)
    plt.show()

.. code:: ipython3

    checkpoint = {
        'hparams': hparams,
        'model_state': model.state_dict(),
        'scaler': scaler,
        'encoder': encoder
    }
    torch.save(checkpoint, 'data/checkpoints/checkpoint.pth')

.. code:: ipython3

    #Path to test file
    fn_wav = 'data/test.wav' # positive example

.. code:: ipython3

    #Inference
    # load model from checkpoint
    loaded_checkpoint = torch.load('data/checkpoints/checkpoint.pth')
    
    hparams = loaded_checkpoint['hparams']
    scaler = loaded_checkpoint['scaler']
    encoder = loaded_checkpoint['encoder']
    
    model = CoughNet(len(hparams['features']))
    model.eval()
    model.load_state_dict(loaded_checkpoint['model_state'])
    
    # create input features
    df_features = pd.DataFrame(columns=hparams['features'])
    df_features = df_features.append(preproces(fn_wav), ignore_index=True)
    X = np.array(df_features[hparams['features']], dtype=np.float32)
    X = torch.Tensor(scaler.transform(X))
    
    outputs = torch.softmax(model(X), 1)
    predictions = torch.argmax(outputs.data, 1)
    
    # print result
    print(f'model outputs {outputs[0].detach().numpy()} which predicts the class {encoder.classes_[predictions]}!')


.. parsed-literal::

    model outputs [9.9903202e-01 9.6800784e-04] which predicts the class covid!
    

.. code:: ipython3

    #k-Fold Cross Validation
    # system imports
    import os
    from datetime import datetime
    
    # additional imports
    import pandas as pd
    import numpy as np
    from tqdm.auto import tqdm
    
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split, KFold
    
    # internal imports
    from utils import plot_confusion_matrix
    # device config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

.. code:: ipython3

    #Hyperparameters
    hparams = {    
        'dataset': 'data/prepared_data_balanced.csv',
        'epochs': 20,
        'batch_size': 16,
        'lr': 1e-3,
        'features': [
            'chroma_stft', 'rmse', 'spectral_centroid', 'spectral_bandwidth', 'rolloff', 'zero_crossing_rate',
            'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 
            'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20'
        ]
    }

.. code:: ipython3

    #Prepare Data
    df_features = pd.read_csv(hparams['dataset'])
    X = np.array(df_features[hparams['features']], dtype=np.float32)
    
    encoder = LabelEncoder()
    y = encoder.fit_transform(df_features['label'])

.. code:: ipython3

    #K-fold Cross Validation model evaluation
    k_folds = 8
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    indices = np.arange(len(y))
    results_train = []
    results_test = []
    
    def train(loader_train, model, optimizer, epoch):
        model.train()
        running_correct = 0.0
        total = 0
        for batch_ndx, sample in enumerate(loader_train): 
            features, labels = sample[0].to(device), sample[1].to(device) 
    
            # forward pass and loss calculation
            outputs = model(features)
            loss = criterion(outputs, labels)  
            
            # backward pass    
            loss.backward()
            
            # update weights
            optimizer.step()
            optimizer.zero_grad()
    
            # calculate metrics
            predictions = torch.argmax(outputs.data, 1)
            running_correct += (predictions == labels).sum().item()
            total += labels.shape[0]
    
        return running_correct / total
    
    def evaluate(loader_test, model, epoch):
        model.eval()
        running_correct = 0.0
        total = 0
        with torch.no_grad():
            for batch_ndx, sample in enumerate(loader_test):
                features, labels = sample[0].to(device), sample[1].to(device) 
    
                # forward pass and loss calculation
                outputs = model(features)
                loss = criterion(outputs, labels)  
    
                # calculate metrics
                predictions = torch.argmax(outputs.data, 1)
                running_correct += (predictions == labels).sum().item()
                total += labels.shape[0]
    
        return running_correct / total
    
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------------------')
    print('|         | Train Accuracy | Test Accuracy |')
    print('--------------------------------------------')
    
    for fold, (train_ids, test_ids) in enumerate(kfold.split(indices)):
        X_train = X[train_ids]
        y_train = y[train_ids]
        X_test = X[test_ids]
        y_test = y[test_ids]
        
        # scale data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # create pytorch dataloader
        torch.manual_seed(42)
        train_dataset = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train).long())
        test_dataset = torch.utils.data.TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test).long())
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=hparams['batch_size'], shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=hparams['batch_size'], shuffle=False)
        
        # create model
        model = CoughNet(len(hparams['features'])).to(device)
    
        # Construct loss and optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
        criterion = torch.nn.CrossEntropyLoss()
    
        # training loop
        for epoch in range(hparams['epochs']):
            train_accuracy = train(train_loader, model, optimizer, epoch)
            eval_accuracy = evaluate(test_loader, model, epoch)
        results_train.append(train_accuracy) 
        results_test.append(eval_accuracy) 
        print(f'| Fold {fold}  |       {train_accuracy*100:.2f} % |       {eval_accuracy*100:.2f} % |')
    
    print('--------------------------------------------')
    print(f'| Average |       {np.mean(results_train)*100:.2f} % |       {np.mean(results_test)*100:.2f} % |')


.. parsed-literal::

    K-FOLD CROSS VALIDATION RESULTS FOR 8 FOLDS
    --------------------------------------------
    |         | Train Accuracy | Test Accuracy |
    --------------------------------------------
    | Fold 0  |       100.00 % |       94.12 % |
    | Fold 1  |       100.00 % |       88.24 % |
    | Fold 2  |       100.00 % |       94.12 % |
    | Fold 3  |       100.00 % |       76.47 % |
    | Fold 4  |       100.00 % |       88.24 % |
    | Fold 5  |       100.00 % |       94.12 % |
    | Fold 6  |       100.00 % |       100.00 % |
    | Fold 7  |       96.61 % |       87.50 % |
    --------------------------------------------
    | Average |       99.58 % |       90.35 % |
    

.. code:: ipython3

    #Training and Evaluation
    #Prepare Data
    df_features = pd.read_csv(hparams['dataset'])
    
    X = np.array(df_features[hparams['features']], dtype=np.float32)
    
    encoder = LabelEncoder()
    y = encoder.fit_transform(df_features['label'])
    print('classes:', encoder.classes_)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print('X_train.shape:', X_train.shape)
    print('y_train.shape:', y_train.shape)


.. parsed-literal::

    classes: ['covid' 'not_covid']
    X_train.shape: (107, 26)
    y_train.shape: (107,)
    

.. code:: ipython3

    def train_eval_classifier(clf):
        clf.fit(X_train, y_train)
    
        predictions = clf.predict(X_train)
        accuracy_train = np.sum(predictions == y_train) / len(y_train)
        print("Train Accuracy:", accuracy_train)
    
        predictions = clf.predict(X_test)
        accuracy_test = np.sum(predictions == y_test) / len(y_test)
        print("Test Accuracy:", accuracy_test)
    
        plot_confusion_matrix(y_test, predictions, encoder.classes_)
    
    def k_fold_train_eval_classifier(clf):    
        k_folds = 4
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        indices = np.arange(len(y))
    
        results_train = []
        results_test = []
    
        print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
        print('--------------------------------------------')
        print('|         | Train Accuracy | Test Accuracy |')
        print('--------------------------------------------')
    
        for fold, (train_ids, test_ids) in enumerate(kfold.split(indices)):
            X_train = X[train_ids]
            y_train = y[train_ids]
            X_test = X[test_ids]
            y_test = y[test_ids]
    
            # train classifier
            clf.fit(X_train, y_train)
    
            # evaluate classifier on train dataset
            predictions = clf.predict(X_train)
            train_accuracy = np.sum(predictions == y_train) / len(y_train)
            results_train.append(train_accuracy) 
    
            # evaluate classifier on test dataset
            predictions = clf.predict(X_test)
            eval_accuracy = np.sum(predictions == y_test) / len(y_test)        
            results_test.append(eval_accuracy) 
    
            print(f'| Fold {fold}  |       {train_accuracy*100:.2f} % |       {eval_accuracy*100:.2f} % |')
    
        print('--------------------------------------------')
        print(f'| Average |       {np.mean(results_train)*100:.2f} % |       {np.mean(results_test)*100:.2f} % |')
        
        plot_confusion_matrix(y_test, predictions, encoder.classes_)

.. code:: ipython3

    #Naive Bayes
    from sklearn.naive_bayes import GaussianNB
    
    clf = GaussianNB()
    k_fold_train_eval_classifier(clf)


.. parsed-literal::

    K-FOLD CROSS VALIDATION RESULTS FOR 4 FOLDS
    --------------------------------------------
    |         | Train Accuracy | Test Accuracy |
    --------------------------------------------
    | Fold 0  |       80.00 % |       70.59 % |
    | Fold 1  |       80.00 % |       61.76 % |
    | Fold 2  |       74.26 % |       87.88 % |
    | Fold 3  |       73.27 % |       81.82 % |
    --------------------------------------------
    | Average |       76.88 % |       75.51 % |
    

.. code:: ipython3

    #Support Verctor Machine
    from sklearn import svm
    
    clf = svm.NuSVC(kernel='poly')
    k_fold_train_eval_classifier(clf)


.. parsed-literal::

    K-FOLD CROSS VALIDATION RESULTS FOR 4 FOLDS
    --------------------------------------------
    |         | Train Accuracy | Test Accuracy |
    --------------------------------------------
    | Fold 0  |       86.00 % |       73.53 % |
    | Fold 1  |       79.00 % |       70.59 % |
    | Fold 2  |       80.20 % |       63.64 % |
    | Fold 3  |       78.22 % |       84.85 % |
    --------------------------------------------
    | Average |       80.85 % |       73.15 % |
    

.. code:: ipython3

    #RandomForestClassifier
    from sklearn.ensemble import RandomForestClassifier
    
    clf = RandomForestClassifier(max_depth=10, n_estimators=100, random_state=42)
    k_fold_train_eval_classifier(clf)


.. parsed-literal::

    K-FOLD CROSS VALIDATION RESULTS FOR 4 FOLDS
    --------------------------------------------
    |         | Train Accuracy | Test Accuracy |
    --------------------------------------------
    | Fold 0  |       100.00 % |       82.35 % |
    | Fold 1  |       100.00 % |       76.47 % |
    | Fold 2  |       100.00 % |       90.91 % |
    | Fold 3  |       100.00 % |       87.88 % |
    --------------------------------------------
    | Average |       100.00 % |       84.40 % |
    

.. code:: ipython3

    #Artificial NeuralNetwork with RELU Activation
    #Extracting the Spectrogram for every Audio File
    #Loading CSV file
    train_csv = pd.read_csv("data/cough_trial_extended.csv")
    train_csv




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>file_properties</th>
          <th>class</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0v8MGxNetjg_ 10.000_ 20.000.wav</td>
          <td>not_covid</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1j1duoxdxBg_ 70.000_ 80.000.wav</td>
          <td>not_covid</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1MSYO4wgiag_ 120.000_ 130.000.wav</td>
          <td>not_covid</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1PajbAKd8Kg_ 0.000_ 10.000.wav</td>
          <td>not_covid</td>
        </tr>
        <tr>
          <th>4</th>
          <td>cov1.wav</td>
          <td>covid</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>165</th>
          <td>-bZrDCS8KAg_ 70.000_ 80.000.wav</td>
          <td>not_covid</td>
        </tr>
        <tr>
          <th>166</th>
          <td>-ej81N6Aqo4_ 0.000_ 8.000.wav</td>
          <td>not_covid</td>
        </tr>
        <tr>
          <th>167</th>
          <td>-gvLnl1smfs_ 90.000_ 100.000.wav</td>
          <td>not_covid</td>
        </tr>
        <tr>
          <th>168</th>
          <td>-hu5q-Nn4BM_ 70.000_ 80.000.wav</td>
          <td>not_covid</td>
        </tr>
        <tr>
          <th>169</th>
          <td>-jLQkyDhIxw_ 10.000_ 20.000.wav</td>
          <td>not_covid</td>
        </tr>
      </tbody>
    </table>
    <p>170 rows × 2 columns</p>
    </div>



.. code:: ipython3

    train_csv['class'].unique()




.. parsed-literal::

    array(['not_covid', 'covid'], dtype=object)



.. code:: ipython3

    cmap = plt.get_cmap('inferno')
    tot_rows = train_csv.shape[0]
    for i in range(tot_rows):
        source = train_csv['file_properties'][i]
        filename = 'data/trial_covid/'+source
        y,sr = librosa.load(filename, mono=True, duration=5)
        plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
        plt.axis('off');
        #plt.savefig(f'./{source[:-3].replace(".", "")}.png')
        #plt.savefig(f'data/plot/{source[:-3].replace(".", "")}.png')
        plt.show()
        #plt.clf()

.. code:: ipython3

    #Extracting features from Spectrogram
    
    #We will extract
    
        #Mel-frequency cepstral coefficients (MFCC)(20 in number)
        #Spectral Centroid,
        #Zero Crossing Rate
        #Chroma Frequencies
        #Spectral Roll-off.
    
    
    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()

.. code:: ipython3

    header




.. parsed-literal::

    ['filename',
     'chroma_stft',
     'rmse',
     'spectral_centroid',
     'spectral_bandwidth',
     'rolloff',
     'zero_crossing_rate',
     'mfcc1',
     'mfcc2',
     'mfcc3',
     'mfcc4',
     'mfcc5',
     'mfcc6',
     'mfcc7',
     'mfcc8',
     'mfcc9',
     'mfcc10',
     'mfcc11',
     'mfcc12',
     'mfcc13',
     'mfcc14',
     'mfcc15',
     'mfcc16',
     'mfcc17',
     'mfcc18',
     'mfcc19',
     'mfcc20',
     'label']



.. code:: ipython3

    #Writing data to csv file
    
    #We write the data to a csv file
    
    file = open('data/data_new_extended.csv', 'w')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
    for i in range(tot_rows):
            source = train_csv['file_properties'][i]
            file_name = 'data/trial_covid/'+source
            y,sr = librosa.load(file_name, mono=True, duration=5)
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            rmse = librosa.feature.rms(y=y)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            to_append = f'{source[:-3].replace(".", "")} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
            for e in mfcc:
                to_append += f' {np.mean(e)}'
            
            file = open('data/data_new_extended.csv', 'a')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())

.. code:: ipython3

    #Analysing the Data in Pandas
    data1 = pd.read_csv('data/data_new_extended.csv')
    data1




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>filename</th>
          <th>chroma_stft</th>
          <th>rmse</th>
          <th>spectral_centroid</th>
          <th>spectral_bandwidth</th>
          <th>rolloff</th>
          <th>zero_crossing_rate</th>
          <th>mfcc1</th>
          <th>mfcc2</th>
          <th>mfcc3</th>
          <th>...</th>
          <th>mfcc12</th>
          <th>mfcc13</th>
          <th>mfcc14</th>
          <th>mfcc15</th>
          <th>mfcc16</th>
          <th>mfcc17</th>
          <th>mfcc18</th>
          <th>mfcc19</th>
          <th>mfcc20</th>
          <th>label</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0v8MGxNetjg_</th>
          <td>10000_</td>
          <td>20000.000000</td>
          <td>0.519951</td>
          <td>0.045853</td>
          <td>1612.895795</td>
          <td>1411.838677</td>
          <td>2907.580566</td>
          <td>0.107019</td>
          <td>-376.876007</td>
          <td>111.017372</td>
          <td>...</td>
          <td>-0.656475</td>
          <td>-7.439712</td>
          <td>-1.034580</td>
          <td>-0.203083</td>
          <td>-3.513495</td>
          <td>-1.745705</td>
          <td>-3.011878</td>
          <td>-2.878482</td>
          <td>-2.106427</td>
          <td>-4.026825</td>
        </tr>
        <tr>
          <th>1j1duoxdxBg_</th>
          <td>70000_</td>
          <td>80000.000000</td>
          <td>0.535472</td>
          <td>0.001771</td>
          <td>2892.087076</td>
          <td>2467.408141</td>
          <td>5072.664388</td>
          <td>0.148584</td>
          <td>-519.158447</td>
          <td>60.781284</td>
          <td>...</td>
          <td>-0.156307</td>
          <td>-0.909973</td>
          <td>7.216461</td>
          <td>-1.719629</td>
          <td>3.903021</td>
          <td>3.653039</td>
          <td>3.043882</td>
          <td>2.439957</td>
          <td>2.781968</td>
          <td>2.195162</td>
        </tr>
        <tr>
          <th>1MSYO4wgiag_</th>
          <td>120000_</td>
          <td>130000.000000</td>
          <td>0.496666</td>
          <td>0.033657</td>
          <td>3429.061935</td>
          <td>2788.634413</td>
          <td>6886.288452</td>
          <td>0.225315</td>
          <td>-282.297913</td>
          <td>48.581680</td>
          <td>...</td>
          <td>0.829615</td>
          <td>-6.066336</td>
          <td>-4.167640</td>
          <td>1.017302</td>
          <td>-0.523806</td>
          <td>0.538693</td>
          <td>-8.855953</td>
          <td>-2.927977</td>
          <td>-1.118562</td>
          <td>-5.906228</td>
        </tr>
        <tr>
          <th>1PajbAKd8Kg_</th>
          <td>0000_</td>
          <td>10000.000000</td>
          <td>0.407549</td>
          <td>0.013452</td>
          <td>2710.811637</td>
          <td>2664.287550</td>
          <td>5778.474935</td>
          <td>0.142076</td>
          <td>-346.857300</td>
          <td>75.765617</td>
          <td>...</td>
          <td>-2.838680</td>
          <td>5.053118</td>
          <td>-0.291308</td>
          <td>0.987186</td>
          <td>-2.447526</td>
          <td>3.692367</td>
          <td>2.312328</td>
          <td>-2.059656</td>
          <td>-4.772599</td>
          <td>-0.503851</td>
        </tr>
        <tr>
          <th>cov1</th>
          <td>0.41269657015800476</td>
          <td>0.059004</td>
          <td>1555.648634</td>
          <td>1418.599932</td>
          <td>2870.737092</td>
          <td>0.133998</td>
          <td>-340.588013</td>
          <td>104.156700</td>
          <td>-32.228443</td>
          <td>-13.615362</td>
          <td>...</td>
          <td>0.940006</td>
          <td>-5.701087</td>
          <td>-6.326630</td>
          <td>-1.080040</td>
          <td>-1.812609</td>
          <td>-2.518986</td>
          <td>-3.684266</td>
          <td>-3.564146</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>-bZrDCS8KAg_</th>
          <td>70000_</td>
          <td>80000.000000</td>
          <td>0.492974</td>
          <td>0.005093</td>
          <td>1600.647469</td>
          <td>2300.999728</td>
          <td>3660.644531</td>
          <td>0.047815</td>
          <td>-543.776917</td>
          <td>119.100296</td>
          <td>...</td>
          <td>-2.675646</td>
          <td>-1.250754</td>
          <td>-2.634280</td>
          <td>1.647435</td>
          <td>0.647164</td>
          <td>1.602689</td>
          <td>-2.469729</td>
          <td>0.704325</td>
          <td>-5.352920</td>
          <td>-1.281080</td>
        </tr>
        <tr>
          <th>-ej81N6Aqo4_</th>
          <td>0000_</td>
          <td>8000.000000</td>
          <td>0.400283</td>
          <td>0.052132</td>
          <td>2664.129566</td>
          <td>2563.440387</td>
          <td>5518.182373</td>
          <td>0.121514</td>
          <td>-290.840607</td>
          <td>85.514404</td>
          <td>...</td>
          <td>-8.843078</td>
          <td>-4.629812</td>
          <td>-7.424622</td>
          <td>-4.511141</td>
          <td>-7.482200</td>
          <td>-4.865530</td>
          <td>-6.353733</td>
          <td>-5.024187</td>
          <td>-8.422812</td>
          <td>-0.831208</td>
        </tr>
        <tr>
          <th>-gvLnl1smfs_</th>
          <td>90000_</td>
          <td>100000.000000</td>
          <td>0.704281</td>
          <td>0.058739</td>
          <td>3090.031219</td>
          <td>2740.856272</td>
          <td>6530.841064</td>
          <td>0.179077</td>
          <td>-75.595451</td>
          <td>68.849228</td>
          <td>...</td>
          <td>-6.867559</td>
          <td>0.677697</td>
          <td>-7.535110</td>
          <td>0.602187</td>
          <td>-6.629556</td>
          <td>0.659050</td>
          <td>-4.125255</td>
          <td>0.734950</td>
          <td>-4.655417</td>
          <td>-0.645009</td>
        </tr>
        <tr>
          <th>-hu5q-Nn4BM_</th>
          <td>70000_</td>
          <td>80000.000000</td>
          <td>0.424896</td>
          <td>0.044159</td>
          <td>3173.872023</td>
          <td>2482.951387</td>
          <td>5768.306478</td>
          <td>0.221743</td>
          <td>-264.064514</td>
          <td>58.729767</td>
          <td>...</td>
          <td>-3.354259</td>
          <td>-0.625627</td>
          <td>0.677355</td>
          <td>-3.651989</td>
          <td>-6.051376</td>
          <td>1.211774</td>
          <td>-14.923816</td>
          <td>-11.180058</td>
          <td>-8.861263</td>
          <td>-5.078876</td>
        </tr>
        <tr>
          <th>-jLQkyDhIxw_</th>
          <td>10000_</td>
          <td>20000.000000</td>
          <td>0.434573</td>
          <td>0.104041</td>
          <td>3006.457898</td>
          <td>2270.008544</td>
          <td>5383.550008</td>
          <td>0.225385</td>
          <td>-113.609337</td>
          <td>61.575642</td>
          <td>...</td>
          <td>-8.916544</td>
          <td>1.918063</td>
          <td>-8.441331</td>
          <td>2.808456</td>
          <td>-6.152548</td>
          <td>-4.181546</td>
          <td>-7.060247</td>
          <td>-0.964895</td>
          <td>0.560492</td>
          <td>-1.245851</td>
        </tr>
      </tbody>
    </table>
    <p>170 rows × 28 columns</p>
    </div>



.. code:: ipython3

    import seaborn as sns
    d = data1.drop(['filename','label'],axis=1)
    h = sns.heatmap(d)

.. code:: ipython3

    data1.shape




.. parsed-literal::

    (170, 28)



.. code:: ipython3

    # Dropping unneccesary columns
    data1 = data1.drop(['filename'],axis=1)

.. code:: ipython3

    #Encoding the Labels
    genre_list = data1.iloc[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(genre_list)

.. code:: ipython3

    #Scaling the Feature columns
    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(data1.iloc[:, :-1], dtype = float))

.. code:: ipython3

    #Dividing data into training and Testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

.. code:: ipython3

    X_train




.. parsed-literal::

    array([[-0.47911629, -0.34810407, -0.35098599, ...,  0.52821075,
             1.16211853, -0.34429923],
           [-0.76024191, -0.34813349, -0.35093973, ..., -0.48967341,
            -0.66474473, -1.2194404 ],
           [ 0.79872742, -0.34802956, -0.35083   , ..., -1.7306292 ,
            -0.09508418, -1.51249787],
           ...,
           [-0.47911629, -0.34815882, -0.35093991, ...,  0.92040063,
            -1.57632789,  1.98086406],
           [ 0.67094305, -0.34780793, -0.35099899, ...,  0.52941914,
             0.52278956,  0.11991464],
           [ 3.22663048, -0.34804996, -0.35081867, ...,  1.02691138,
             0.92830204, -0.08233566]])



.. code:: ipython3

    #Classification with Keras
    #Building our Network
    
    import tensorflow as tf
    from tensorflow import keras
    from keras import models
    from keras import layers
    from keras.layers import Dropout
    from keras.utils.vis_utils import plot_model
    
    
    model = tf.keras.Sequential()
    model.add(layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.3, input_shape=(60,)))
    
    model.add(layers.Dense(64, activation='relu'))
    
    #model.add(layers.Dense(128, activation='relu'))
    
    #model.add(layers.Dense(64, activation='relu'))
    
    model.add(layers.Dense(10, activation='relu'))
    
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # serialize model to JSON
    #model_json = model.to_json()
    #with open("model.json", "w") as json_file:
        #json_file.write(model_json)
    # serialize weights to HDF5
    #model.save_weights("model.h5")
    print("Saved model to disk")
    
    # plot model
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


.. parsed-literal::

    Saved model to disk
    ('You must install pydot (`pip install pydot`) and install graphviz (see instructions at https://graphviz.gitlab.io/download/) ', 'for plot_model/model_to_dot to work.')
    

.. code:: ipython3

    model.summary()


.. parsed-literal::

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_4 (Dense)              (None, 128)               3456      
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 128)               0         
    _________________________________________________________________
    dense_5 (Dense)              (None, 64)                8256      
    _________________________________________________________________
    dense_6 (Dense)              (None, 10)                650       
    _________________________________________________________________
    dense_7 (Dense)              (None, 1)                 11        
    =================================================================
    Total params: 12,373
    Trainable params: 12,373
    Non-trainable params: 0
    _________________________________________________________________
    

.. code:: ipython3

    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=100, batch_size=128)


.. parsed-literal::

    Epoch 1/100
    2/2 [==============================] - 1s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 2/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 3/100
    2/2 [==============================] - 0s 2ms/step - loss: nan - accuracy: 0.0074
    Epoch 4/100
    2/2 [==============================] - 0s 2ms/step - loss: nan - accuracy: 0.0074
    Epoch 5/100
    2/2 [==============================] - 0s 5ms/step - loss: nan - accuracy: 0.0074
    Epoch 6/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 7/100
    2/2 [==============================] - 0s 6ms/step - loss: nan - accuracy: 0.0074
    Epoch 8/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 9/100
    2/2 [==============================] - 0s 5ms/step - loss: nan - accuracy: 0.0074
    Epoch 10/100
    2/2 [==============================] - 0s 2ms/step - loss: nan - accuracy: 0.0074
    Epoch 11/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 12/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 13/100
    2/2 [==============================] - 0s 2ms/step - loss: nan - accuracy: 0.0074
    Epoch 14/100
    2/2 [==============================] - 0s 4ms/step - loss: nan - accuracy: 0.0074
    Epoch 15/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 16/100
    2/2 [==============================] - 0s 5ms/step - loss: nan - accuracy: 0.0074
    Epoch 17/100
    2/2 [==============================] - 0s 2ms/step - loss: nan - accuracy: 0.0074
    Epoch 18/100
    2/2 [==============================] - 0s 4ms/step - loss: nan - accuracy: 0.0074
    Epoch 19/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 20/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 21/100
    2/2 [==============================] - 0s 5ms/step - loss: nan - accuracy: 0.0074
    Epoch 22/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 23/100
    2/2 [==============================] - 0s 4ms/step - loss: nan - accuracy: 0.0074
    Epoch 24/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 25/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 26/100
    2/2 [==============================] - 0s 4ms/step - loss: nan - accuracy: 0.0074
    Epoch 27/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 28/100
    2/2 [==============================] - 0s 5ms/step - loss: nan - accuracy: 0.0074
    Epoch 29/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 30/100
    2/2 [==============================] - 0s 4ms/step - loss: nan - accuracy: 0.0074
    Epoch 31/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 32/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 33/100
    2/2 [==============================] - 0s 4ms/step - loss: nan - accuracy: 0.0074
    Epoch 34/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 35/100
    2/2 [==============================] - 0s 5ms/step - loss: nan - accuracy: 0.0074
    Epoch 36/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 37/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 38/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 39/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 40/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 41/100
    2/2 [==============================] - 0s 4ms/step - loss: nan - accuracy: 0.0074
    Epoch 42/100
    2/2 [==============================] - 0s 5ms/step - loss: nan - accuracy: 0.0074
    Epoch 43/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 44/100
    2/2 [==============================] - 0s 4ms/step - loss: nan - accuracy: 0.0074
    Epoch 45/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 46/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 47/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 48/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 49/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 50/100
    2/2 [==============================] - 0s 4ms/step - loss: nan - accuracy: 0.0074
    Epoch 51/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 52/100
    2/2 [==============================] - 0s 4ms/step - loss: nan - accuracy: 0.0074
    Epoch 53/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 54/100
    2/2 [==============================] - 0s 4ms/step - loss: nan - accuracy: 0.0074
    Epoch 55/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 56/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 57/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 58/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 59/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 60/100
    2/2 [==============================] - 0s 2ms/step - loss: nan - accuracy: 0.0074
    Epoch 61/100
    2/2 [==============================] - 0s 2ms/step - loss: nan - accuracy: 0.0074
    Epoch 62/100
    2/2 [==============================] - 0s 2ms/step - loss: nan - accuracy: 0.0074
    Epoch 63/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 64/100
    2/2 [==============================] - 0s 2ms/step - loss: nan - accuracy: 0.0074
    Epoch 65/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 66/100
    2/2 [==============================] - 0s 2ms/step - loss: nan - accuracy: 0.0074
    Epoch 67/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 68/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 69/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 70/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 71/100
    2/2 [==============================] - 0s 4ms/step - loss: nan - accuracy: 0.0074
    Epoch 72/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 73/100
    2/2 [==============================] - 0s 4ms/step - loss: nan - accuracy: 0.0074
    Epoch 74/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 75/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 76/100
    2/2 [==============================] - 0s 2ms/step - loss: nan - accuracy: 0.0074
    Epoch 77/100
    2/2 [==============================] - 0s 5ms/step - loss: nan - accuracy: 0.0074
    Epoch 78/100
    2/2 [==============================] - 0s 2ms/step - loss: nan - accuracy: 0.0074
    Epoch 79/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 80/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 81/100
    2/2 [==============================] - 0s 4ms/step - loss: nan - accuracy: 0.0074
    Epoch 82/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 83/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 84/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 85/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 86/100
    2/2 [==============================] - 0s 2ms/step - loss: nan - accuracy: 0.0074
    Epoch 87/100
    2/2 [==============================] - 0s 2ms/step - loss: nan - accuracy: 0.0074
    Epoch 88/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 89/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 90/100
    2/2 [==============================] - 0s 2ms/step - loss: nan - accuracy: 0.0074
    Epoch 91/100
    2/2 [==============================] - 0s 2ms/step - loss: nan - accuracy: 0.0074
    Epoch 92/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 93/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 94/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 95/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 96/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 97/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 98/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    Epoch 99/100
    2/2 [==============================] - 0s 2ms/step - loss: nan - accuracy: 0.0074
    Epoch 100/100
    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0074
    

.. code:: ipython3

    test_loss, test_acc = model.evaluate(X_test,y_test)


.. parsed-literal::

    2/2 [==============================] - 0s 3ms/step - loss: nan - accuracy: 0.0000e+00
    

.. code:: ipython3

    print('test_acc: ',test_acc)


.. parsed-literal::

    test_acc:  0.0
    

.. code:: ipython3

    #Predictions on Test Data
    predictions = model.predict(X_test)
    print(predictions[0].shape)
    print(np.sum(predictions[0]))
    print(predictions[:4])
    print(y_test[:4])


.. parsed-literal::

    (1,)
    nan
    [[nan]
     [nan]
     [nan]
     [nan]]
    [ 87  93 121 107]
    

.. code:: ipython3

    #Saving the Spectograms as a single output file
    !tar -zcvf outputname.tar.zip /data/plot


.. parsed-literal::

    tar: : Couldn't visit directory: No such file or directory
    tar: Error exit delayed from previous errors.
    

.. code:: ipython3

    df = pd.DataFrame(predictions, columns = ['Negative','Positive'])


::


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_15916\3230673770.py in <module>
    ----> 1 df = pd.DataFrame(predictions, columns = ['Negative','Positive'])
    

    ~\AppData\Roaming\Python\Python39\site-packages\pandas\core\frame.py in __init__(self, data, index, columns, dtype, copy)
        670                 )
        671             else:
    --> 672                 mgr = ndarray_to_mgr(
        673                     data,
        674                     index,
    

    ~\AppData\Roaming\Python\Python39\site-packages\pandas\core\internals\construction.py in ndarray_to_mgr(values, index, columns, dtype, copy, typ)
        322     )
        323 
    --> 324     _check_values_indices_shape_match(values, index, columns)
        325 
        326     if typ == "array":
    

    ~\AppData\Roaming\Python\Python39\site-packages\pandas\core\internals\construction.py in _check_values_indices_shape_match(values, index, columns)
        391         passed = values.shape
        392         implied = (len(index), len(columns))
    --> 393         raise ValueError(f"Shape of passed values is {passed}, indices imply {implied}")
        394 
        395 
    

    ValueError: Shape of passed values is (34, 1), indices imply (34, 2)


