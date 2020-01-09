import os
import time
from pathlib import Path

import torch
from torch import nn, optim
import cv2
from config import params
from lib import slowfastnet

import numpy as np
from torch.utils.data import DataLoader, Dataset, globFile

import numpy as np



class VideoDataset(Dataset):
    def __init__(self, directory, mode='train', clip_len=8, frame_sample_rate=1):
        #global list_index
        folder = Path(directory) / mode  # get the directory of the specified split
        self.clip_len = clip_len
        self.folder = folder
        self.short_side = [128, 160]
        self.crop_size = 112
        self.frame_sample_rate = frame_sample_rate
        self.mode = mode
        self.list_index = globFile.list_index

        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(label)
        # prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        label_file = str(len(os.listdir(folder))) + 'class_labels.txt'
        with open(label_file, 'w') as f:
            for id, label in enumerate(sorted(self.label2index)):
                f.writelines(str(id + 1) + ' ' + label + '\n')

    def __getitem__(self, index):
        # loading and preprocessing. TODO move them to transform classes
        buffer = self.loadvideo(self.fnames[index])

        while buffer.shape[0] < self.clip_len + 2:
            index = np.random.randint(self.__len__())
            buffer = self.loadvideo(self.fnames[index])

        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        label_file = 'index.txt'

        with open(label_file, 'a') as f:
            f.write(str(index) + '\n')
            f.close()

        #print(index)
        #globFile.list_index = self.list_index.append(index)
        return buffer, self.label_array[index]

    def to_tensor(self, buffer):
        # convert from [D, H, W, C] format to [C, D, H, W] (what PyTorch uses)
        # D = Depth (in this case, time), H = Height, W = Width, C = Channels
        return buffer.transpose((3, 0, 1, 2))

    def loadvideo(self, fname):
        remainder = np.random.randint(self.frame_sample_rate)
        # initialize a VideoCapture object to read video data into a numpy array
        capture = cv2.VideoCapture(fname)

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if frame_height < frame_width:
            resize_height = np.random.randint(self.short_side[0], self.short_side[1] + 1)
            resize_width = int(float(resize_height) / frame_height * frame_width)
        else:
            resize_width = np.random.randint(self.short_side[0], self.short_side[1] + 1)
            resize_height = int(float(resize_width) / frame_width * frame_height)

        # create a buffer. Must have dtype float, so it gets converted to a FloatTensor by Pytorch later
        start_idx = 0
        end_idx = frame_count - 1
        frame_count_sample = frame_count // self.frame_sample_rate - 1
        if frame_count > 300:
            end_idx = np.random.randint(300, frame_count)
            start_idx = end_idx - 300
            frame_count_sample = 301 // self.frame_sample_rate - 1
        buffer = np.empty((frame_count_sample, resize_height, resize_width, 3), np.dtype('float32'))

        count = 0
        retaining = True
        sample_count = 0

        # read in each frame, one at a time into the numpy buffer array
        while (count <= end_idx and retaining):
            retaining, frame = capture.read()
            if count < start_idx:
                count += 1
                continue
            if retaining is False or count > end_idx:
                break
            if count % self.frame_sample_rate == remainder and sample_count < frame_count_sample:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # will resize frames if not already final size

                if (frame_height != resize_height) or (frame_width != resize_width):
                    frame = cv2.resize(frame, (resize_width, resize_height))
                buffer[sample_count] = frame
                sample_count = sample_count + 1
            count += 1
        capture.release()
        return buffer

    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        time_index = np.random.randint(buffer.shape[0] - clip_len)
        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        # crop and jitter the video using indexing. The spatial crop is performed on 
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer

    def normalize(self, buffer):
        # Normalize the buffer
        # buffer = (buffer - 128)/128.0
        for i, frame in enumerate(buffer):
            frame = (frame - np.array([[[128.0, 128.0, 128.0]]])) / 128.0
            buffer[i] = frame
        return buffer

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""
        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer

    def __len__(self):
        return len(self.fnames)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, correct_k


def main():
    '''
    torch.backends.cudnn.benchmark = True
    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    logdir = os.path.join(params['log'], cur_time)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    '''
    #writer = SummaryWriter(log_dir=logdir)

    #videos = videos_reales()

    with open('index.txt', 'w') as f:
        f.close()

    print("Loading dataset")

    data = VideoDataset(params['dataset'], mode='validation', clip_len=64, frame_sample_rate=1)
    #print('hola')
    val_dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=8)

    print("load model")
    model = slowfastnet.resnet50(class_num=params['num_classes'])

    if params['pretrained'] is not None:
        pretrained_dict = torch.load(params['pretrained'], map_location='cpu')
        try:
            model_dict = model.module.state_dict()
        except AttributeError:
            model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        print("load pretrain model")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model = model.cuda(params['gpu'][0])
    model = nn.DataParallel(model, device_ids=params['gpu'])  # multi-Gpu

    model.eval()

    #contadores agregados para guardar video
    n = 0
    m = 0

    #esto se puede generalizar recorriendo una lista de las clases y haciendo un for despues de otro
    dic_predict_1 = dict()
    dic_predict_1['cellphone'] = 0
    dic_predict_1['nothing'] = 0
    dic_predict_2 = dict()
    dic_predict_2['cellphone'] = 0
    dic_predict_2['nothing'] = 0
    predicciones = dict()
    predicciones['cellphone'] = dic_predict_1
    predicciones['nothing'] = dic_predict_2
    end = time.time()
    with torch.no_grad():
        for step, (inputs, labels) in enumerate(val_dataloader):
            #data_time.update(time.time() - end)
            #largo batch, canales, numero de imagenes, (112, 112)

            with open('index.txt', 'r') as f:
                list_index = f.readlines()
                f.close()
            labels_2 = labels
            inputs_cpu = np.array(inputs)
            inputs = inputs.cuda()
            #largo 16 por que hay 16 videos por batch
            labels = labels.cuda()
            #videos,clases 16,101, filas,columnas
            outputs = model(inputs)

            res, correct_k = accuracy(outputs.data, labels)

            video_path = data.fnames[int(list_index[n].rstrip('\n'))]

            capture = cv2.VideoCapture(video_path)

            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

            out_np = outputs.cpu().detach().numpy()
            index = np.argmax(out_np)
            '''
            if (int(correct_k.cpu()/100) != (index == np.array(labels_2)[0])):
                print('ESTA MAL')
            print(int(correct_k.cpu() / 100), index, np.array(labels_2)[0])
            '''
            #ESTO DEJA AL DICCIONARIO COMO predicciones[lo que realemente es][lo que penso que era]
            #se puede mejorar leyendo el archivo class_labels.txt para hacerlo general
            if index == 0:
                texto = 'cellphone'
                if np.array(labels_2) == 0:
                    predicciones['cellphone']['cellphone'] += 1
                else:
                    predicciones['nothing']['cellphone'] += 1
            else:
                texto = 'nothing'
                if np.array(labels_2) == 0:
                    predicciones['cellphone']['nothing'] += 1
                else:
                    predicciones['nothing']['nothing'] += 1
            #print(list_index, outputs)
            #modificacion para guardar ciertos videos de la validacion y escribir lo que infiere la red

            vid_writer = cv2.VideoWriter('videos_out/video_' + str(n) + '.avi', cv2.VideoWriter_fourcc(*'MPEG'), 20, (frame_width, frame_height), True)
            while capture.isOpened():
                ret, imag = capture.read()
                if imag is None:
                    break
                cv2.putText(imag, texto, (frame_width - 200, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                vid_writer.write(imag.astype(np.uint8))
            capture.release()
            n += 1
    print(predicciones)

if __name__ == '__main__':
    main()
