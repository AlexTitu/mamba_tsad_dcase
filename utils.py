from torch.utils.data import Dataset
import torch
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# -- Dataloader concentrated on the whole dataset --
# --------------------------------------------------
class DCASE2024Dataset(Dataset):
  def __init__(self, root_dir, dataset, transform=None, extension='.wav', standardize=False):
    self.root_dir = root_dir
    self.transform = transform
    self.datasetType = dataset
    self.standardize = standardize
    self.mean = None
    self.std = None
    self.datasetSource = []
    self.datasetTarget = []
    self.labels = []
    self.extension = extension
    self.load_dataset()

    if self.standardize and self.mean is None and self.std is None:
      self.compute_statistics()

  def __len__(self):
    return len(self.datasetSource) + len(self.datasetTarget)

  def sourceLen(self):
    return len(self.datasetSource)

  def targetLen(self):
    return len(self.datasetTarget)

  def compute_statistics(self):
    all_specs = []
    for audioPath in self.datasetSource + self.datasetTarget:
      if self.extension == '.npy':
        audioFile = np.load(audioPath)
      else:
        audioFile = self.logmelspec(audioPath, 'statistics')
      all_specs.append(audioFile)

    all_specs = np.concatenate(all_specs, axis=0)
    self.mean = np.mean(all_specs, axis=0)
    self.std = np.std(all_specs, axis=0)

  def load_dataset(self):
    # first, we have the instance_machine folders
    machineData = os.listdir(self.root_dir)
    for index, machineFolder in enumerate(machineData):
        folderType, machineType = machineFolder.split("_")
        if folderType == self.datasetType[0]:
            machineSounds = os.path.join(self.root_dir, machineFolder, machineType, self.datasetType[1])
            for recording in os.listdir(machineSounds):
                # Check if the file is a .npy file
                if recording.endswith(self.extension):
                    self.labels.append(index)
                    if recording.find('source') != -1:
                        self.datasetSource.append(os.path.join(machineSounds, recording))
                    elif recording.find('target') != -1:
                        self.datasetTarget.append(os.path.join(machineSounds, recording))
        else:
            continue

  def preGenerateMelSpecs(self):
    if self.extension == '.wav':
      for audioPath in self.datasetSource + self.datasetTarget:
          if not os.path.exists(os.path.splitext(audioPath)[0]+'.npy'):
            mel_spec = self.logmelspec(audioPath, 'preGenMelSpecs')
            # Get the directory and filename without the extension
            directory, filename = os.path.split(os.path.splitext(audioPath)[0])
            # Save the Mel spectrogram as a numpy array in the same directory as the audio file
            np.save(os.path.join(directory, filename + '.npy'), mel_spec)
          else:
            continue
    else:
      print(f'Pre-Generation of Mel Specs is not supported!')
      return

    return

  def shapeof(self, index):
      sample, _ = self.__getitem__(index)
      return np.shape(sample)

  def logmelspec(self, path, origin):
    audioFile, sr = librosa.load(path, sr=16000)

    # Ensure audio is 10 seconds long
    if len(audioFile) < sr * 10:
        audioFile = np.pad(audioFile, (0, sr * 10 - len(audioFile)), constant_values=(0, 0))
    else:
        audioFile = audioFile[:sr * 10]

    # Compute the Mel spectrogram with a fixed number of Mel frequency bands and a fixed hop length
    audiomelspec = librosa.feature.melspectrogram(y=audioFile, sr=sr, n_fft=1024, hop_length=512, n_mels=128) # 1251 / 5049
    audiomelspec_db = librosa.power_to_db(audiomelspec)

    if self.standardize and origin != 'statistics':
      audiomelspec_db = (audiomelspec_db - self.mean) / self.std

    audiomelspec_db = np.array([audiomelspec_db])

    return audiomelspec_db

  def __getitem__(self, index):
    if index >= len(self.datasetSource):
      index = len(self.datasetSource)-index
      audioPath = self.datasetTarget[index]
    else:
      audioPath = self.datasetSource[index]

    if self.extension == '.npy':
      audioFile = np.load(audioPath)
    else:
      audioFile = self.logmelspec(audioPath, 'getitem')

    if self.transform:
      audioFile = self.transform(audioFile)

    if audioPath.find('anomaly') != -1:
      return audioFile, 1
    else:
      return audioFile, 0


# -------------------------------------------------
# -- Dataloader concentrated on the machine type --
# -------------------------------------------------
class DCASE2024MachineDataset(Dataset):
  def __init__(self, root_dir, dataset, machine, transform=None, extension='.wav', standardize=None, isTesting=False):
    self.root_dir = root_dir
    self.transform = transform
    self.datasetType = dataset
    self.machineType = machine.split("_")[1]
    self.isTesting = isTesting

    valid = {'statistical', 'min-max', None}
    if standardize not in valid:
      raise ValueError("Error: standardize must be one of %r." % valid)
    else:
      self.standardize = standardize

    self.mean = None
    self.std = None
    self.datasetSource = []
    self.datasetTarget = []
    self.extension = extension
    self.load_dataset()

    if self.standardize == 'statistical' and self.mean is None and self.std is None:
      self.compute_statistics()

  def __len__(self):
    return len(self.datasetSource) + len(self.datasetTarget)

  def sourceLen(self):
    return len(self.datasetSource)

  def targetLen(self):
    return len(self.datasetTarget)

  def compute_statistics(self):
    all_specs = []
    for audioPath in self.datasetSource + self.datasetTarget:
      if self.extension == '.npy':
        audioFile = np.load(audioPath)
      else:
        audioFile = self.logmelspec(audioPath, 'statistics')
      all_specs.append(audioFile)

    all_specs = np.concatenate(all_specs, axis=0)
    self.mean = np.mean(all_specs, axis=0)
    self.std = np.std(all_specs, axis=0)

  def load_dataset(self):
    # first, we have the instance_machine folders
    machineData = os.listdir(self.root_dir)
    for index, machineFolder in enumerate(machineData):
        folderType, machineType = machineFolder.split("_")
        if folderType == self.datasetType[0] and machineType == self.machineType:
            machineSounds = os.path.join(self.root_dir, machineFolder, machineType, self.datasetType[1])
            for recording in os.listdir(machineSounds):
                # Check if the file is a .npy file
                if recording.endswith(self.extension):
                    if recording.find('source') != -1:
                        self.datasetSource.append(os.path.join(machineSounds, recording))
                    elif recording.find('target') != -1:
                        self.datasetTarget.append(os.path.join(machineSounds, recording))

        else:
            continue

  def preGenerateMelSpecs(self, override=False):
    if self.extension == '.wav':
      for audioPath in self.datasetSource + self.datasetTarget:
          if not os.path.exists(os.path.splitext(audioPath)[0]+'.npy') or override:
            mel_spec = self.logmelspec(audioPath, 'preGenMelSpecs')
            # Get the directory and filename without the extension
            directory, filename = os.path.split(os.path.splitext(audioPath)[0])
            # Save the Mel spectrogram as a numpy array in the same directory as the audio file
            np.save(os.path.join(directory, filename + '.npy'), mel_spec)
          else:
            continue
    else:
      print(f'Pre-Generation of Mel Specs is not supported!')
      return

    return

  def shapeof(self, index):
      sample, _ = self.__getitem__(index)
      return np.shape(sample)

  def logmelspec(self, path, origin):
    audioFile, sr = librosa.load(path, sr=16000)

    # Ensure audio is 10 seconds long
    if len(audioFile) < sr * 10:
        audioFile = np.pad(audioFile, (0, sr * 10 - len(audioFile)), constant_values=(0, 0))
    else:
        audioFile = audioFile[:sr * 10]

    # Compute the Mel spectrogram with a fixed number of Mel frequency bands and a fixed hop length
    audiomelspec = librosa.feature.melspectrogram(y=audioFile, sr=sr, n_fft=1024,
                                                  hop_length=512, n_mels=128) # 1251 / 5049
    audiomelspec_db = librosa.power_to_db(audiomelspec)

    if self.standardize == 'statistical' and origin != 'statistics':
        audiomelspec_db = (audiomelspec_db - self.mean) / self.std

    elif self.standardize == 'min-max':
        audiomelspec_db = ((audiomelspec_db - np.min(audiomelspec_db)) /
                           (np.max(audiomelspec_db) - np.min(audiomelspec_db)))

    audiomelspec_db = np.array([audiomelspec_db])

    return audiomelspec_db

  def __getitem__(self, index):
    if index >= len(self.datasetSource):
      origin = 'target'
      index = len(self.datasetSource)-index
      audioPath = self.datasetTarget[index]
    else:
      origin = 'source'
      audioPath = self.datasetSource[index]

    if self.extension == '.npy':
      audioFile = np.load(audioPath)
    else:
      audioFile = self.logmelspec(audioPath, 'getitem')

    if self.transform:
      audioFile = self.transform(audioFile)

    if not self.isTesting:
        if audioPath.find('anomaly') != -1:
          return audioFile, 1
        else:
          return audioFile, 0
    else:
        if audioPath.find('anomaly') != -1:
            return audioFile, 1, origin
        else:
            return audioFile, 0, origin


# -----------------------------------
# -- Defining Early Stopping Class --
# -----------------------------------
class EarlyStopping():
  def __init__(self, patience=1, min_delta=0, models_dir='./models_dir'):
    self.patience = patience
    self.min_delta = min_delta
    self.counter = 0
    self.models_dir = models_dir
    self.min_validation_loss = np.inf
    self.best_epoch = 0
    self.best_train = None
    self.best_val = None

  def earlyStop(self, epoch, trainLoss, valLoss, model):
    if valLoss <= (self.min_validation_loss + self.min_delta):
      print("[INFO] In EPOCH {} the loss value improved from {:.5f} to {:.5f}".format(epoch, self.min_validation_loss, valLoss))
      self.setMinValLoss(valLoss)
      self.setCounter(0)
      self.setBestEpoch(epoch)
      torch.save(model.state_dict(), f"{self.models_dir}/CAE_normed.pt")
      self.setBestLosses(trainLoss, valLoss)

    elif valLoss > (self.min_validation_loss + self.min_delta):
      self.setCounter(self.counter + 1)
      print("[INFO] In EPOCH {} the loss value did not improve from {:.5f}. This is the {} EPOCH in a row.".format(epoch, self.min_validation_loss, self.counter))
      if self.counter >= self.patience:
        return True
    return False

  def setCounter(self, counter_state):
    self.counter = counter_state

  def setMinValLoss(self, ValLoss):
    self.min_validation_loss = ValLoss

  def setBestLosses(self, trainLoss, valLoss):
    self.best_train = trainLoss
    self.best_val = valLoss

  def setBestEpoch(self, bestEpoch):
    self.best_epoch = bestEpoch

  def getBestTrainLoss(self):
    return self.best_train

  def getBestValLoss(self):
    return self.best_val

  def getBestEpoch(self):
    return self.best_epoch

  def saveLossesLocally(self):
    np.save(f'{self.models_dir}/losses_train_normed.npy', np.array(self.best_train))
    np.save(f'{self.models_dir}/losses_val_normed.npy', np.array(self.best_val))

  def loadLossesLocally(self):
    self.best_train = np.load(f'{self.models_dir}/losses_train_normed.npy')
    self.best_val = np.load(f'{self.models_dir}/losses_val_normed.npy')


# -------------------------------------
# -- Printing and plotting functions --
# -------------------------------------
def plotSpectrogram(spectrogram, audioType):
    input_shape = spectrogram.shape
    print(spectrogram.shape)
    print(f'Audio type: {audioType}')
    fig, ax = plt.subplots()
    img = librosa.display.specshow(spectrogram[0], x_axis='time',
                           y_axis='mel', sr=16000,
                           fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.show()


# ensuring even split of classes in train and validation sets
def stratified_split(dataset, validation_size=0.25, random_state=None):
    # Assuming 'dataset.labels' contains your labels
    labels = dataset.labels

    # Create stratified split for test set
    train_index, val_index = train_test_split(
        range(len(labels)),
        test_size=validation_size,
        random_state=random_state,
        stratify=labels)

    return train_index, val_index


def printDataInfo(dataset):
    for spectrogram, tag in dataset:
        print(np.shape(spectrogram))
        print(np.min(spectrogram))
        print(np.max(spectrogram))
        print(tag)
        break


"""
  # switching off autograd for eval
        with torch.no_grad():
          # set the model in eval mode
          model.eval()
          trainLosses = []

          for mel_specs, _ in trainDataLoader:
            mel_specs = mel_specs.to(device)

            pred_mel_specs = model(mel_specs)
            loss_val = lossFn(pred_mel_specs, mel_specs)
            trainLosses.append(loss_val.cpu().detach().item())

        # Assume trainLosses is a list of the reconstruction errors on your training data
        train_losses = np.array(trainLosses)
        mean = np.mean(train_losses)
        std_dev = np.std(train_losses)
        print(f"Total loss mean: {mean}")
        print(f"Total loss std: {std_dev}")

        # Set threshold as mean plus 3 standard deviations
        threshold = mean + 3 * std_dev

        H['threshold'] = threshold

        torch.save({
          'train_loss_history': H}, f"{models_dir}/train_history_values_normed.pt")

"""
