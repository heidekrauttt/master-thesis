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
<img src="https://github.com/heidekrauttt/master-thesis/blob/main/results/corpus_analysis/percentage_dialogue_over_time.png" alt="dialogue-over-time" width="600"/>
<img src="https://github.com/heidekrauttt/master-thesis/blob/main/results/corpus_analysis/absolute_dialogue_over_time.png" alt="absolute-dialogue" width="600"/>


#### Word Clouds
Find [all wordclouds here](https://github.com/heidekrauttt/master-thesis/tree/main/results/corpus_analysis/wordclouds). <br>
<img src="https://github.com/heidekrauttt/master-thesis/blob/main/results/corpus_analysis/wordclouds/wordcloud-fem.png" alt="wordcloud-fem" width="600"/> <br>


#### Standardized Type Token Ratio
<img src="https://github.com/heidekrauttt/master-thesis/blob/main/results/corpus_analysis/sttr_boxplot-allgenders.png" alt="STTR" width="600"/> <br>


### Sentiment Analysis
A sentiment analysis was conducted using [this transformer model](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli). <br>
The analysis used the labels 'Positive', 'Neutral' and 'Negative. Goal was to investigate differences between the speech patterns between the three gender groups female, male, and diverse. <br>

Plots were created for each game, as the following example shows. These plots can also be found [here](https://github.com/heidekrauttt/master-thesis/tree/main/results/sentiment-analysis/plots/allgenders_sentiment).

<img src="https://github.com/heidekrauttt/master-thesis/blob/main/results/sentiment-analysis/plots/allgenders_sentiment/sentiment_plot_kingsquest5.png" alt="sentiment analysis" width="500"> <br>
Separate results for the sentiments of the three groups can be found [here](https://github.com/heidekrauttt/master-thesis/tree/main/results/sentiment-analysis/plots/separate-analysis-plots). The results are also saved as [.csv files](https://github.com/heidekrauttt/master-thesis/tree/main/results/sentiment-analysis/csv-files).

### Emotion Detection

#### Base Emotions
The labelset consisted of the labels:  <br>
<img src="https://github.com/heidekrauttt/master-thesis/blob/main/results/emotion-detection/plots/base-emotion-utterance-wise/without-surprise-label/combined_plots/all_games_mean_percentage_barplot.png" alt="Base emotions percentage" width="500"><br>
<img src="https://github.com/heidekrauttt/master-thesis/blob/main/results/emotion-detection/plots/base-emotion-utterance-wise/without-surprise-label/combined_plots/all_games_mean_scores_barplot.png" alt="Base emotions mean" width="500"><br>


#### Experimental Labelset 1
The labelset consisted of the labels:  <br>
<img src="https://github.com/heidekrauttt/master-thesis/blob/main/results/emotion-detection/plots/experimental-utterance-wise-labelset-1/all-gendergroups-compared/all_games_mean_percentage_barplot.png" alt="Experimental labelset 1 percentage" width="500"><br>
<img src="https://github.com/heidekrauttt/master-thesis/blob/main/results/emotion-detection/plots/experimental-utterance-wise-labelset-1/all-gendergroups-compared/all_games_mean_scores_barplot.png" alt="Experimental labelset 1 mean" width="500"><br>


#### Experimental Labelset 2
The labelset consisted of the labels:  <br>
<img src="https://github.com/heidekrauttt/master-thesis/blob/main/results/emotion-detection/plots/experimental-utterance-wise-labelset-2/all-gendergroups-compared/all_games_mean_percentage_barplot.png" alt="Experimental labelset 2 percentage" width="500"><br>
<img src="https://github.com/heidekrauttt/master-thesis/blob/main/results/emotion-detection/plots/experimental-utterance-wise-labelset-2/all-gendergroups-compared/all_games_mean_scores_barplot.png" alt="Experimental labelset 2 mean" width="500"><br>


## Results
The video games in the present corpus show clear differences between male characters and female and diverse characters in terms of dialogue. Investigating all present dialogue for the three gender groups over 19 games, male characters dominated games both in number and in dialogue lines. Masculine hegemony is portrayed by the male characters of the games through the statistical overpowering, establishing a dominance through presence and representation. Male characters show different lexical features than other characters, and use different vocabulary. The wordcloud analysis revealed that male characters exhibit linguistic habits that are in line with the more traditional, binary theories of masculinity, when transmitted to speech patterns. The overrepresentation of male characters and the omnipresence of male characters holds for all target groups, and is especially pronounced for the 'Teen' games, painting a concerning image on diverse representation in these games. The structures exhibited in the corpus are in line with the hegemonic framework of masculinities proposed by R.W. Connell: the close reading showed that male characters are in relation with one another and explicitly discuss hierarchies regarding their status, in contrast to female and diverse characters. <br>
Unfortunately, the diverse characters in the corpus make up only a small amount of the dialogue and there are few characters. All statistical values were therefore low. However, the diverse characters do not seem to follow conventional character design for males or females. They exhibit different vocabulary and take on different roles, expressing different emotions in different games. <br>
Female and diverse characters do not yet have an equal share of characters and storylines dedicated to them in video games. Fortunately, the industry is changing rapidly and many new games are released per year. One popular example is the successor of 'Hades', the game 'Hades II' was released in 2024 and features 'Melinoë' as a main character. She is the sister of the first game's main playable character 'Zagreus', showcasing how game series can impact representation of different characters in new game installations. Making non-male characters autonomous main characters in games is one way to increase visibility, change current dynamics in gaming, and work against the dominance of masculinity. <br>
