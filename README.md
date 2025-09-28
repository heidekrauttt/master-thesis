# Master Thesis
Open repository containing the code for my Master thesis in Digital Humanities at the University of Stuttgart. <br>

## Abstract
The digital world is ever expanding, its realms stretch into more and more areas of our daily lives. Consumers of digital media value the immersion, the break from reality, that these digitized worlds are able to provide. However, even in the digital, we cannot escape all of the implicit structures of our society. In this thesis, the influence of gender on video games will be investigated. Incorporating gender theories and game studies, a corpus of video game dialogue will be analyzed and parallels will be drawn to the real world. Using a mixture of established corpus linguistics methods, emotion and sentiment analysis with a state-of-the-art transformer model, and a close reading approach, the characters' performance of gender through their speech will be evaluated. 
The study shows that characters of different genders behave differently, choosing different vocabulary and displaying an array of emotions and sentiments. Overall, male characters dominate the corpus, across subsections such as subgenre or target group. Most games have some diverse character representation, however it is marginal in comparison.
The findings of this study will provide a solid foundation for further research and will offer valuable insights into the portrayal of masculinity and diverse characters in video games.

## Methods
In order to assess masculinities in game dialogue, the master thesis implemented multiple methods. 
[This script](https://github.com/heidekrauttt/master-thesis/blob/main/scripts/run_basic_analysis.py) runs the complete analysis, including most of the plots. Further plots can be created using the [postprocessing script](https://github.com/heidekrauttt/master-thesis/blob/main/scripts/postprocessing.py). <br>
Some example results are shown in the following.

### Corpus Analysis
#### Word Clouds
#### Standardized Type Token Ratio
#### Basic Statistics
### Sentiment Analysis
A sentiment analysis was conducted using [this transformer model](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli). <br>
The analysis used the labels 'Positive', 'Neutral' and 'Negative. Goal was to investigate differences between the speech patterns between the three gender groups female, male, and diverse. <br>

Plots were created for each game, as the following example shows. These plots can also be found [here](https://github.com/heidekrauttt/master-thesis/tree/main/results/sentiment-analysis/plots/allgenders_sentiment).

[<img src="https://github.com/heidekrauttt/master-thesis/blob/main/results/sentiment-analysis/plots/allgenders_sentiment/sentiment_plot_kingsquest5.png">] <br>
Separate results for the sentiments of the three groups can be found [here](https://github.com/heidekrauttt/master-thesis/tree/main/results/sentiment-analysis/plots/separate-analysis-plots). The results are also saved as [.csv files](https://github.com/heidekrauttt/master-thesis/tree/main/results/sentiment-analysis/csv-files).
### Emotion Detection
#### Base Emotions
The labelset consisted of the labels:
#### Experimental Labelset 1
The labelset consisted of the labels:
#### Experimental Labelset 2
The labelset consisted of the labels:
## Results
