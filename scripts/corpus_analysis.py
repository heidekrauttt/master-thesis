import math
import os
import statistics
import string
import time
from collections import Counter

import pandas as pd
import nltk
# only uncomment on first iteration
# nltk.download('punkt_tab')
import re
from statistics import mean
from matplotlib import pyplot as plt
from wordcloud import WordCloud


def basic_stats(gendered_dict_of_dialogue):
    '''
    A method that calculates basic statistics
    # use the dialogue corpora by gender to get stats about dialogue
    # add all lines of dialogue to one big string, regardless of speaker
    :parameter  gendered_dict_of_dialogue - a dictionary containing all dialogue we have extracted from one gender
    :returns:   long_gendered_dict_of_dialogue - a dictionary with basic statistics
                stats - a pandas dataframe of statistics

    author: @heidekrauttt
    '''

    all_lines = ''
    length_dialogue_sum = []
    gamevariants = []
    length_dialogue = []
    unique_names = []
    long_gendered_dict_of_dialogue = {}
    for gamevariant, dict_of_dialogue in gendered_dict_of_dialogue.items():
        dialogue_df = gendered_dict_of_dialogue[gamevariant]
        lines = len(dialogue_df['Dialogue'])
        num_characters = len(set(dialogue_df['Name']))
        for line in dialogue_df['Dialogue']:
            all_lines = all_lines + line

        #dialogue_df['length_dialogue'] = dialogue_df['Dialogue'].apply(lambda x: len(nltk.word_tokenize(x, language="english")))
        dialogue_df['length_dialogue'] = dialogue_df['Dialogue'].apply(
            lambda x: len([re.sub(r'[^\w\s]', '', token) for token in nltk.word_tokenize(x, language="english") if
                           re.sub(r'[^\w\s]', '', token)]))

        length_dialogue_sum.append(dialogue_df['length_dialogue'].sum())
        gamevariants.append(gamevariant)
        #gamevariants += len(dialogue_df['Dialogue']) * [gamevariant]
        unique_names.append(len(set(dialogue_df['Name'])))
        # dialogue_df['unique_names'] = dialogue_df['Name'].apply(lambda x: len(set(dialogue_df['Name'])))
        # dialogue_df['sum_length_dialogue'] = dialogue_df['length_dialogue'].sum()
        long_gendered_dict_of_dialogue[gamevariant] = dialogue_df

        # create dict from info
    stats = {'gamevariant': gamevariants, 'num_characters': unique_names, 'length_dialogue_sum': length_dialogue_sum}
    stats = pd.DataFrame(stats)

    return long_gendered_dict_of_dialogue, stats


def tokenize(dialogue_df):
    '''
    Tokenizes a given dialogue in the provided dataframe. Removes punctuation.
    :param      dialogue_df - dataframe containing the dialogue
    :return:    tokenized_dialogue - a tokenized version of the dialogue, without punctuation
                all_lines - a long string containing all lines of dialogue without punctuation

    author: @heidekrauttt
    '''
    lines = len(dialogue_df['Dialogue'])
    all_lines = []
    tokenized_dialogue_no_punct = dialogue_df['Dialogue'].apply(
        lambda x: [re.sub(r'[^\w\s]', '', token) for token in nltk.word_tokenize(x, language="english") if
                   re.sub(r'[^\w\s]', '', token)])
    for line in tokenized_dialogue_no_punct:
        for token in line:
            all_lines.append(token.lower())  # lowercase version
    return tokenized_dialogue_no_punct, all_lines


def tokenize_with_punctuation(dialogue_df):
    '''
    Tokenizes a given dialogue in the provided dataframe. Keeps punctuation.
    :param dialogue_df: dataframe containing the dialogue
    :return:    tokenized_dialogue - a tokenized version of the dialogue.
                all_lines - a long string containing all lines of dialogue.

    author: @heidekrauttt
    '''
    all_lines = []
    tokenized_dialogue = dialogue_df['Dialogue'].apply(lambda x: (nltk.word_tokenize(x, language="english")))
    for line in tokenized_dialogue:
        for token in line:
            all_lines.append(token.lower())  # lowercase version
    return tokenized_dialogue, all_lines


def ttr_per_game(fem_dict, male_dict, div_dict):
    '''
    Method that calculates the type token ratio per game. TTR is a method to measure lexical richness.
    :param fem_dict - dictionary of all female dialogue
    :param male_dict - dictionary of all male dialogue
    :param div_dict - dictionary of all diverse dialogue
    :return: null; plots the ttr for female, male and diverse tokens

    author: @heidekrauttt
    '''
    gamevariants = []
    female_ttrs = []
    male_ttrs = []
    div_ttrs = []
    overall_ttrs = []

    # for each game (fem is used for iteration)
    for gamevariant, df_of_dialogue in fem_dict.items():
        # get all female dialogue for the game
        _, female = tokenize(df_of_dialogue)
        # get all male dialogue for the game
        _, male = tokenize(male_dict.get(gamevariant))
        # get all diverse dialogue for the game
        _, div = tokenize(div_dict.get(gamevariant))
        # calculate ttrs

        if (len(female) + len(male) + len(div)) < 1:
            continue  # error catcher
        if len(female) > 0:
            female_ttr = len(set(female)) / len(female)
        else:
            female_ttr = 0.0

        if len(male) > 0:
            male_ttr = len(set(male)) / len(male)
        else:
            male_ttr = 0.0
        if len(div) > 0:
            div_ttr = len(set(div)) / len(div)
        else:
            div_ttr = 0.0

        overall_ttr = (len(set(female)) + len(set(male)) + len(set(div))) / (len(female) + len(male) + len(div))

        female_ttrs.append(female_ttr)
        male_ttrs.append(male_ttr)
        div_ttrs.append(div_ttr)
        overall_ttrs.append(overall_ttr)
        gamevariants.append(gamevariant)
    ttrs = {'female': female_ttrs, 'male': male_ttrs, 'div': div_ttrs, 'gamevariant': gamevariants}
    ttrs = pd.DataFrame(ttrs)
    ttrs.plot(x='gamevariant', style='o')

def sttr_per_game(fem_dict, male_dict, div_dict):
    '''
    Method that calculates the standardized type token ratio per game.
    STTR is a method to measure lexical richness.
    :param fem_dict - dictionary of all female dialogue
    :param male_dict - dictionary of all male dialogue
    :param div_dict - dictionary of all diverse dialogue
    :return: null; plots the sttr for female, male and diverse tokens

    author: @heidekrauttt
    '''
    gamevariants = []
    female_ttrs = []
    male_ttrs = []
    div_ttrs = []
    overall_ttrs = []

    # for each game (fem is used for iteration)
    for gamevariant, df_of_dialogue in fem_dict.items():
        # get all female dialogue for the game
        _, female = tokenize(df_of_dialogue)
        # get all male dialogue for the game
        _, male = tokenize(male_dict.get(gamevariant))
        # get all diverse dialogue for the game
        _, div = tokenize(div_dict.get(gamevariant))
        # calculate sttrs

        if (len(female) + len(male) + len(div)) < 1:
            continue  # error catcher
        if len(female) > 0:
            female_ttr = len(set(female)) / math.sqrt(len(female) * 2)
        else:
            female_ttr = 0.0

        if len(male) > 0:
            male_ttr = len(set(male)) / math.sqrt(len(male) * 2)
        else:
            male_ttr = 0.0
        if len(div) > 0:
            div_ttr = len(set(div)) / math.sqrt(len(div) * 2)
        else:
            div_ttr = 0.0

        overall_ttr = (len(set(female)) + len(set(male)) + len(set(div))) / math.sqrt((len(female) + len(male) + len(div)) * 2)

        female_ttrs.append(female_ttr)
        male_ttrs.append(male_ttr)
        div_ttrs.append(div_ttr)
        overall_ttrs.append(overall_ttr)
        gamevariants.append(gamevariant)

    # calculate sttr
    data = {
        'TTR': female_ttrs + male_ttrs + div_ttrs,
        'Gender': ['female'] * len(female_ttrs) + ['male'] * len(male_ttrs) + ['diverse'] * len(div_ttrs)
    }
    sttrs_df = pd.DataFrame(data)

    # Create the box plot using pandas
    plt.figure(figsize=(8, 6))
    sttrs_df.boxplot(column='TTR', by='Gender', grid=False, rot=0)
    plt.title('STTR Distribution by Gender')
    plt.suptitle('')  # Suppress the default title generated by pandas
    plt.xlabel('Gender Group')
    plt.ylabel('STTR')
    plt.savefig('sttr_boxplot.png')
    plt.show()


def mean_sentence_length(fem_dict, male_dict, div_dict):
    mean_f = []
    mean_m = []
    mean_d = []
    for gamevariant, df_of_dialogue in fem_dict.items():
        # get all female dialogue for the game
        _, female = tokenize_with_punctuation(df_of_dialogue)
        # get all male dialogue for the game
        _, male = tokenize_with_punctuation(male_dict.get(gamevariant))
        # get all diverse dialogue for the game
        _, div = tokenize_with_punctuation(div_dict.get(gamevariant))

        # sentence tokenize
        sent_length_f = []
        tokenized_dialogue_f = df_of_dialogue['Dialogue'].apply(lambda x: (nltk.sent_tokenize(x, language="english")))

        sent_length_m = []
        tokenized_dialogue_m = male_dict.get(gamevariant)['Dialogue'].apply(
            lambda x: (nltk.sent_tokenize(x, language="english")))

        sent_length_d = []
        tokenized_dialogue_d = div_dict.get(gamevariant)['Dialogue'].apply(
            lambda x: (nltk.sent_tokenize(x, language="english")))
        # get all sentence lengths
        for line in tokenized_dialogue_f:
            for entry in line:
                sent_length_f.append(len(entry.split(" ")))
        if len(sent_length_f) > 0:
            mean_f.append(mean(sent_length_f))
        else:
            mean_f.append(0)

        for line in tokenized_dialogue_m:
            for entry in line:
                sent_length_m.append(len(entry.split(" ")))
        if len(sent_length_m) > 0:
            mean_m.append(mean(sent_length_m))
        else:
            mean_m.append(0)

        for line in tokenized_dialogue_d:
            for entry in line:
                sent_length_d.append(len(entry.split(" ")))
        if len(sent_length_d) > 0:
            mean_d.append(mean(sent_length_d))
        else:
            mean_d.append(0)

    print(statistics.mean(mean_f))
    print(statistics.mean(mean_m))
    print(statistics.mean(mean_d))
    means = pd.DataFrame({'female': mean_f, 'male': mean_m, 'diverse': mean_d})

def most_frequent_words_onegender(input_folder, output_path, wordclass="all"):
    # hab ich schon irgendwo alle worte in einem riesen string? dann koennte ich Counter() nutzen
    all_texts = ""
    # get stopwords
    stopwords = set(nltk.corpus.stopwords.words('english'))
    # get punctuation to exclude
    punctuation = set(string.punctuation)
    # append other frequent punctuations that are not in this set
    punctuation.update(['.', '..', '...', ',', '--', '\'\'', '\"\"', '\'', '\`'])
    # get all txt files
    for file in os.listdir(input_folder):
        start = time.time()
        #gamevariant = str(file).replace(".txt", "").split("_")[-1]

        if file.endswith(".txt"):
            all_outputs = [] # Reset for each file
            with open(os.path.join(input_folder, file), "r") as f:
                text = f.read()
                all_texts = all_texts + text
    # remove all stopwords
    all_tokens = nltk.word_tokenize(all_texts)
    filtered_texts = [w for w in all_tokens if not w.lower() in stopwords and not w.lower() in punctuation]
    if wordclass == "verbs":
        #nltk.download('averaged_perceptron_tagger_eng')
        verbs = nltk.pos_tag(filtered_texts)
        verbs = filter(lambda w: w[1] == "VB", verbs)
        filtered_texts = [word for word, pos in verbs]
    elif wordclass == "adjectives":
        #nltk.download('averaged_perceptron_tagger_eng')
        adj = nltk.pos_tag(filtered_texts)
        adj = filter(lambda w: w[1] == "JJ", adj)
        filtered_texts = [word for word, pos in adj]
    # Counter() by gender
    stats = Counter(filtered_texts)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(stats)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plot_filename = output_path
    plt.savefig(plot_filename)
    return stats.most_common(150)

def most_frequent_words(input_fem, input_male, input_div):
    # all words
    stats_fem = most_frequent_words_onegender(input_fem, "results/wordcloud-fem")
    stats_male = most_frequent_words_onegender(input_male, "results/wordcloud-male")
    stats_div = most_frequent_words_onegender(input_div, "results/wordcloud-div")

    # verbs
    stats_fem = most_frequent_words_onegender(input_fem, "results/wordcloud-fem-verbs", wordclass="verbs")
    stats_male = most_frequent_words_onegender(input_male, "results/wordcloud-male-verbs", wordclass="verbs")
    stats_div = most_frequent_words_onegender(input_div, "results/wordcloud-div-verbs", wordclass="verbs")

    # adjectives
    stats_fem = most_frequent_words_onegender(input_fem, "results/wordcloud-fem-adj", wordclass="adjectives")
    stats_male = most_frequent_words_onegender(input_male, "results/wordcloud-male-adj", wordclass="adjectives")
    stats_div = most_frequent_words_onegender(input_div, "results/wordcloud-div-adj", wordclass="adjectives")

def dialogue_over_time(fem_dict, male_dict, div_dict, release_years):
    # for each gamevariant, get release year
    # get all dialogue of one year (for all genders)
    # plot by year (absolute numbers)

    # get total amount of dialogue per year, and percentagewise how much f/m/d have spoken in that year
    # plot by year
    all_dialogue_data = []

    # Iterate through each gamevariant and its year
    for gamevariant, year in release_years.items():
        # Retrieve DataFrames for the current gamevariant
        current_fem_df = fem_dict.get(gamevariant)
        current_male_df = male_dict.get(gamevariant)
        current_div_df = div_dict.get(gamevariant)

        # Ensure DataFrames exist and add 'Year' and 'Gender' if not already present
        # (though they should be from the previous step)
        if current_fem_df is not None and 'Year' not in current_fem_df.columns:
            current_fem_df['Year'] = year
        if current_fem_df is not None and 'Gender' not in current_fem_df.columns:
            current_fem_df['Gender'] = 'female'

        if current_male_df is not None and 'Year' not in current_male_df.columns:
            current_male_df['Year'] = year
        if current_male_df is not None and 'Gender' not in current_male_df.columns:
            current_male_df['Gender'] = 'male'

        if current_div_df is not None and 'Year' not in current_div_df.columns:
            current_div_df['Year'] = year
        if current_div_df is not None and 'Gender' not in current_div_df.columns:
            current_div_df['Gender'] = 'diverse'

        # Concatenate valid DataFrames for the current gamevariant
        game_dialogues = []
        if current_fem_df is not None:
            game_dialogues.append(current_fem_df)
        if current_male_df is not None:
            game_dialogues.append(current_male_df)
        if current_div_df is not None:
            game_dialogues.append(current_div_df)

        if game_dialogues:
            combined_game_df = pd.concat(game_dialogues, ignore_index=True)
            all_dialogue_data.append(combined_game_df)

    if not all_dialogue_data:
        print("No dialogue data found for the given gamevariants and years.")
        return pd.DataFrame(), pd.DataFrame()

    # Combine all dialogue data into a single DataFrame
    full_dialogue_df = pd.concat(all_dialogue_data, ignore_index=True)

    # Calculate absolute dialogue counts per year for each gender
    # We count the number of rows (dialogues) for each gender per year
    absolute_counts = full_dialogue_df.groupby(['Year', 'Gender']).size().unstack(fill_value=0)
    absolute_counts.columns.name = None  # Remove the 'Gender' name from columns index

    # Calculate total dialogue per year
    total_dialogue_per_year = absolute_counts.sum(axis=1)

    # Calculate percentage of dialogue per year for each gender
    percentage_counts = absolute_counts.div(total_dialogue_per_year, axis=0) * 100
    percentage_counts = percentage_counts.fillna(0)  # Fill NaN with 0 if a gender had no dialogue in a year
    # ------------------ Plotting Section ------------------

    # Plot and save absolute counts
    plt.figure(figsize=(10, 6))
    absolute_counts.plot(kind='bar', rot=45)
    plt.title('Absolute Dialogue Counts Per Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Dialogue Lines')
    plt.legend(title='Gender')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('absolute_dialogue_over_time.png')
    plt.show()

    # Plot and save percentage counts
    plt.figure(figsize=(10, 6))
    percentage_counts.plot(kind='bar', stacked=True, rot=45)
    plt.title('Percentage Dialogue Share Per Year')
    plt.xlabel('Year')
    plt.ylabel('Percentage of Dialogue (%)')
    plt.legend(title='Gender')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('percentage_dialogue_over_time.png')
    plt.show()

    # ------------------------------------------------------

    return absolute_counts, percentage_counts