# Video Captioning
Automatically generating description for video content has been an important task in the field of Computer Vision. Since the input is a sequence of video frames, and the caption output is a sequence of words, sequence learning models that are wisely used in machine translation task can also be used for this task.

This repo implements a sequence learning model (encoder-decoder) for video captioning using PyTorch framework. The encoder uses [VGG-16](https://arxiv.org/abs/1409.1556) model pretrained on the [ILSVRC-2012-CLS](http://www.image-net.org/challenges/LSVRC/2012/) dataset for extracting visual feature of video frames. Both the encoder and the decoder are Gated Recurrent Unit (GRU) networks.

![alt text](png/seq2seq.png)
![alt text](png/1.png)
![alt text](png/2.png)

#### Training time


#### Test time




## Usage


#### 1. Download the dataset

[CSV file](https://www.microsoft.com/en-us/download/details.aspx?id=52422&from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fdownloads%2F38cf15fd-b8df-477e-a4e4-a4680caa75af%2Fdefault.aspx) that contains multilingual descriptions of the videos.

[Video files (.avi)](http://www.cs.utexas.edu/users/ml/clamp/videoDescription/YouTubeClips.tar)

#### 2. Preprocessing

Read CSV file into text file, select one clean English caption per video

```bash
$ python read_csv.py
```

Extract video tarball file into the same folder. Then sample all video into frames

```bash
$ python extract_frame.py
```

#### 3. Train the model

```bash
$ python train.py
```

#### 4. Test the model

```bash
$ python sample.py --image='png/example.png'
```

<br>

## Pretrained model
If you do not want to train the model from scratch, you can continue training by using pretrained model: [encoder.pt](https://www.dropbox.com/s/16t3us2yc94ah3p/encoder.pt) [decoder.pt](https://www.dropbox.com/s/zw6428yu8wjh8zf/decoder.pt)

```bash
$ python train_cont.py
```