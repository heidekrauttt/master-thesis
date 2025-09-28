'''
This file is used to run a basic analysis of the corpus at hand.
It is an exemplary usage of the code in the provided files.
Running these scripts will result in a data analysis of game dialogue, split into corpus analysis,
sentiment analysis and emotion detection using a transformer model.
The author is always the owner of this repository, unless otherwise stated.
author: @heidekrauttt
'''
from pathlib import Path

# --- 1. read in corpus folder ---
from analysis_scripts.corpus_analysis import sttr_per_game
from helper_methods import process_folder

# the data folder can't be published due to copyright. please refer to the corpus repository for acquirance.
df_games, games_list, fem, male, div, release_years = process_folder(Path('/data'))

# --- 2. get overview over present data ---
from corpus_analysis import dialogue_over_time
a, p = dialogue_over_time(fem, male, div, release_years)
from helper_methods import plot_dialogue_trends
#plot_dialogue_trends(a, p)

# --- 3. calculate basic statistics ---
from corpus_analysis import basic_stats
from helper_methods import plot_stats

#basic statistics: use the dialogue corpora by gender to get stats about dialogue
analysis_fem, stats_fem = basic_stats(fem)
analysis_male, stats_male = basic_stats(male)
analysis_div, stats_div = basic_stats(div)
# save the results for further analysis
stats_fem.to_csv("results/stats_fem.csv")
stats_male.to_csv("results/stats_male.csv")
stats_div.to_csv("results/stats_div.csv")

# plot the statistics
plot_stats(stats_fem, stats_male, stats_div)

#save dialogue as one big string in a txt file
from helper_methods import process_df_to_txt
process_df_to_txt(analysis_male, "../data/txt_files-male/all_male_dialogue")
process_df_to_txt(analysis_fem, "../data/txt_files-fem/all_female_dialogue")
process_df_to_txt(analysis_div, "../data/txt_files-div/all_diverse_dialogue")

# --- 4. facilitate corpus analysis ---
from corpus_analysis import ttr_per_game, mean_sentence_length
from corpus_analysis import most_frequent_words

# create word clouds
most_frequent_words("../data/txt_files-fem/", "../data/txt_files-male/", "../data/txt_files-div/")

# type token ratio
ttr_per_game(analysis_fem, analysis_male, analysis_div)
# standardized type token ratio
sttr_per_game(analysis_fem, analysis_male, analysis_div)

# mean sentence length
mean_sentence_length(analysis_fem, analysis_male, analysis_div)

# --- 5. run sentiment analysis ---
from sentiment_analysis import sentiment_analysis, emotion_analysis, plot_sentiments, plot_two_sentiments, emotion_analysis_experimental
sentiment_analysis("female", input_folder="../data/txt_files-fem/", output_folder="results/sentiment_analysis", csv_filename="sentiment_analysis-results", plot=True)
sentiment_analysis("male", input_folder="../data/txt_files-male/", output_folder="results/sentiment_analysis", csv_filename="sentiment_analysis-results", plot=True)
sentiment_analysis("diverse", input_folder="../data/txt_files-div/", output_folder="results/sentiment_analysis", csv_filename="sentiment_analysis-results", plot=True)

# --- 6. run emotion detection ---
emotion_analysis("male", input_folder="../data/txt_files-male/", output_folder="results/emotion_analysis-utterancewise-nosurprise", csv_filename="emotion_analysis-utterancewise-nosurprise-results", plot=True)
emotion_analysis("female", input_folder="../data/txt_files-fem/", output_folder="results/emotion_analysis-utterancewise-nosurprise", csv_filename="emotion_analysis-utterancewise-nosurprise-results", plot=True)
emotion_analysis("diverse", input_folder="../data/txt_files-div/", output_folder="results/emotion_analysis-utterancewise-nosurprise", csv_filename="emotion_analysis-utterancewise-nosurprise-results", plot=True)

# --- 7. run experimental approach (manually select labelset) ---
emotion_analysis_experimental("male", input_folder="../data/txt_files-male/", output_folder="results/emotion_analysis-utterancewise-experimental", csv_filename="emotion_analysis-utterancewise-experimental", plot=True)
emotion_analysis_experimental("female", input_folder="../data/txt_files-fem/", output_folder="results/emotion_analysis-utterancewise-experimental", csv_filename="emotion_analysis-utterancewise-experimental", plot=True)
emotion_analysis_experimental("diverse", input_folder="../data/txt_files-div/", output_folder="results/emotion_analysis-utterancewise-experimental", csv_filename="emotion_analysis-utterancewise-experimental", plot=True)

# --- 8. plot if needed ---
plot_sentiments("sentiment_analysis_results-female.csv", "sentiment_analysis_results-male.csv", "sentiment_analysis_results-diverse.csv", output_dir="sentiment_plots/fmd")