import csv
import glob
import re

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def videoExist(Id):
    for filename in sorted(glob.glob('Caption_Video/YouTubeClips/*.avi'), key=numericalSort): #assuming gif
        if str(filename).startswith('Caption_Video/YouTubeClips/' + Id):
            return True
    return False

file = open('corpus.txt', 'w')

prev_id = []
prev_id.append('hi')
with open('corpus.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        vid_id = row['VideoID']
        if (row['Language'] == 'English') and (row['Source'] == 'clean') and (prev_id[0]!=vid_id):
            if videoExist(vid_id):
                file.write(row['VideoID'] + '_' + row['Start'] + '_' + row['End'] + '\t'+ row['Description'] +'\n')
                prev_id.pop()
                prev_id.append(vid_id)
                print(vid_id)

file.close()