import os
import nltk
import argparse
from lexrank import STOPWORDS, LexRank
from path import Path
from autocorrect import spell
words = set(nltk.corpus.words.words())


def scene_detection(video_file, frame_change_rate):
    if not 0.0 < frame_change_rate < 1.0:
        raise ValueError('frame change rate must between 0 and 1')
    print('Running scene detection...')
    if os.path.exists('ffout'):
        os.remove('ffout')
    command = 'ffmpeg -i ' + video_file + ' -filter:v "select=\'gt(scene,' + str(frame_change_rate) \
              + ')\',showinfo" -f null - 2> ffout'
    os.system(command)


def split_video_by_scene(scene_interval):
    raw_time_stamp = []     # time stamps of scene change
    with open('ffout', 'r') as f:
        for line in f:
            if 'pts_time:' in line:
                string_start = line.find('pts_time:') + 9
                string_end = string_start + line[string_start:].find(' ')
                # skip head of video
                if float(line[string_start: string_end]) >= 1.0:
                    raw_time_stamp.append(float(line[string_start: string_end]))
            # get video file name
            if 'from' in line:
                video_file = line[line.find('from') + 6:line.find('\':')]
                if '/' in video_file:
                    video_file_name = video_file[video_file.rfind('/') + 1:-4]
                else:
                    video_file_name = video_file[:-4]

    if len(raw_time_stamp) <= 2:
        raise ValueError('No scene change detected!')
    else:
        print('Scene detection done!')
        print('Splitting video...')

    time_stamp = []     # start/end time of scenes
    for i in range(1, len(raw_time_stamp)):
        if raw_time_stamp[i] - raw_time_stamp[i - 1] >= scene_interval:
            time_stamp.append([raw_time_stamp[i - 1], raw_time_stamp[i]])

    if not os.path.exists('output_video'):
        os.mkdir('output_video')
    else:
        os.system('rm -rf output_video')
        os.mkdir('output_video')

    # split video
    for chunk in time_stamp:
        command = 'ffmpeg -ss ' + str(chunk[0]) + ' -i ' + video_file + ' -c copy -t ' \
                  + str(chunk[1] - chunk[0]) + ' ./output_video/' + video_file_name + '_' \
                  + str(time_stamp.index(chunk)).zfill(3) + '.mp4'
        os.system(command)
    os.remove('ffout')
    print('Video split done!')


def extract_audio_5s(video_file):
    """
    Extract audio in 5 seconds period for DeepSpeech Speech-to-text translation.
    :param video_file:
    :return: None
    """
    if not os.path.exists('output_audio'):
        os.mkdir('output_audio')
    command = 'ffmpeg -i output_video/' + video_file + ' -ac 1 -ar 16000 -acodec pcm_s16le -f segment -segment_time 5 ' + 'output_audio/' + video_file[:-4] + '_%03d.wav'
    os.system(command)


def deepspeech_batch_files_macOS(model, alphabet, lm, trie):
    print('Running speech-to-text translation...')
    if not os.path.exists('output_text'):
        os.mkdir('output_text')

    for audio_file in sorted(os.listdir('output_audio')):
        if audio_file[-3:] == 'wav':
            command = 'deepspeech' + ' ' + model + ' ./output_audio/' + audio_file + ' ' + alphabet + ' ' \
                      + lm + ' ' + trie + ' >> ./output_text/' + audio_file[:-8] + '.txt'
            os.system(command)

    os.system('rm -rf output_audio')


def summarize(text_file, cat, summary_size, threshold):
    if cat not in ['business', 'entertainment', 'politics', 'sport', 'tech']:
        raise ValueError('category must be one of business, entertainment, politics, sport, tech')
    if summary_size <= 0:
        raise ValueError('number of summary sentences must be greater than one')
    if not 0.0 < threshold < 1.0:
        raise ValueError('summarize threshold must between 0 and 1')
    # load parsing dataset
    documents = []
    documents_dir = Path('bbc/' + cat)
    for file_path in documents_dir.files('*.txt'):
        with file_path.open(mode='rt', encoding='utf-8') as fp:
            documents.append(fp.readlines())

    # initialize LexRank with dataset
    lxr = LexRank(documents, stopwords=STOPWORDS['en'])

    # Read STT result
    sentences = []
    with open(text_file, 'r') as f:
        sentences.extend(f.read().splitlines())

    # Spelling correction
    for index in range(len(sentences)):
        tmp = ''
        for word in sentences[index].split():
            tmp += spell(word)
            tmp += ' '
        sentences[index] = tmp

    # Non-English word removal
    for index in range(len(sentences)):
        sentences[index] = " ".join(
            w for w in nltk.wordpunct_tokenize(sentences[index]) if w.lower() in words or not w.isalpha())

    # STT translation after correction
    with open('output_text/' + text_file[:-4] + '_corrected.txt', 'a', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(sentence + '\n')

    # get summary with classical LexRank algorithm
    summary = lxr.get_summary(sentences, summary_size, threshold)
    with open('output_text/' + text_file[:-4] + '_sum.txt', 'a', encoding='utf-8') as f:
        for sentence in summary:
            f.write(sentence + '\n')


def pipeline_process(video_file, model, alphabet, lm, trie,
                     frame_change_rate, scene_interval, cat, summary_size, threshold):
    scene_detection(video_file, frame_change_rate)
    split_video_by_scene(scene_interval)
    for video_chunk in sorted(os.listdir('output_video')):
        if video_chunk[-3:] == 'mp4':
            extract_audio_5s(video_chunk)
            deepspeech_batch_files_macOS(model, alphabet, lm, trie)
    print('All speech-to-text translation done!')
    print('Running text summarization...')
    for text_chunk in sorted(os.listdir('output_text')):
        if text_chunk[-3:] == 'txt':
            summarize(text_chunk, cat, summary_size, threshold)
    print('All process done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Video summarization.')
    parser.add_argument('v',
                        help='video input file')
    parser.add_argument('-f', type=float,
                        help='frame_change_rate', default=0.5)
    parser.add_argument('-i', type=int,
                        help='scene_interval in second', default=400)
    parser.add_argument('m',
                        help='DeepSpeech model')
    parser.add_argument('a',
                        help='DeepSpeech alphabet file')
    parser.add_argument('-lm',
                        help='DeepSpeech lm file', default='')
    parser.add_argument('-trie',
                        help='DeepSpeech trie file', default='')
    parser.add_argument('-c',
                        help='video category(business, entertainment, politics, sport, tech)', default='politics')
    parser.add_argument('-s', type=int,
                        help='number of summary sentences', default=5)
    parser.add_argument('-t', type=float,
                        help='summary threshold', default=0.1)
    args = parser.parse_args()
    video_file = args.v
    frame_change_rate = args.f
    scene_interval = args.i
    model = args.m
    alphabet = args.a
    lm = args.lm
    trie = args.trie
    cat = args.c
    summary_size = args.s
    threshold = args.t
    pipeline_process(video_file, model, alphabet, lm, trie,
                     frame_change_rate, scene_interval, cat, summary_size, threshold)
