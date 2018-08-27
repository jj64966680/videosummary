# Video Summary

This repo contains python code for video summarization. You would get video chunks devided by change of scene, speech-to-text translation, improved translation result and finally the summarization, which can be used as category tag, keyword extraction, etc.

## Dependencies

[NLTK](https://www.nltk.org/) <br />
[ffmpeg](https://www.ffmpeg.org/) <br />
[LexRank](https://github.com/wikibusiness/lexrank) <br />
[autocorrect](https://github.com/phatpiglet/autocorrect) <br />
[DeepSpeech](https://github.com/mozilla/DeepSpeech) <br />

## Instruction

Copy the script under the LexRank folder(same level of bbc folder in the LexRank folder), then try with default settings by running the following command:
```
python3 video_sum.py video_file.mp4 output_graph.pb alphabet.txt
```
where ```output_graph.pb``` and ```alphabet.txt``` are the speech-to-text model and alphabet database of DeepSpeech.<br />

Result would be in the ```output_video``` and ```output_text``` folder under the LexRank folder.<br />

To get more information on arguments, run the following command:
```
python3 video_sum.py -h
```
