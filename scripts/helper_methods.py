import json
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns


def process_folder(data_folder):
    '''
    :param data_folder: path to the data folder, containing results from corpus scraper scripts
    :return:    df_dict - dictionary, containing dataframes with dialogue data
                games_list - list, containing all game names
                all_dialogues_fem - dictionary, complete dialogue by gender (female)
                all_dialogues_male - dictionary, complete dialogue by gender (male)
                all_dialogues_div - dictionary, complete dialogue by gender (diverse)
                release_years - dictionary, release years of games, per game variant

    author: @heidekrauttt
    '''
    all_dialogues = {}
    all_dialogues_fem = {}
    all_dialogues_male = {}
    all_dialogues_div = {}
    release_years = {}

    # list all folders in data folder
    for gamefolder in os.listdir(data_folder):
        gamefolder_path = os.path.join(data_folder, gamefolder)
        # if the gamefolder is a directory (it is indeed a game ordner)
        if os.path.isdir(gamefolder_path):

            # list all folders in gamefolder
            for gamevariant in os.listdir(gamefolder_path):
            # access game variant folder
                gamevariant_path = os.path.join(gamefolder_path, gamevariant)
                # open the json file path
                json_file_path = os.path.join(gamevariant_path, "data.json")

                # save all text lines
                if os.path.exists(json_file_path):
                    with open(json_file_path, "r") as f:
                        json_data = json.load(f)

                    dialogue_lines = extract_dialogue(json_data.get("text", []))
                    all_dialogues[gamevariant] = dialogue_lines

                    # get meta data about gender
                    meta_path = os.path.join(gamevariant_path, "meta.json")
                    if os.path.exists(meta_path):
                        with open(meta_path, "r") as f:
                            # saves all metadata in a dict format so we can access by key later
                            json_meta = json.load(f)

                        game_name = json_meta.get("game", [])
                        char_groups = json_meta.get("characterGroups", [])
                        mpc = json_meta.get("mainPlayerCharacters", [])
                        year = json_meta.get("year", [])
                        release_years[gamevariant] = year
                        status = json_meta.get("status", [])
                        # exclude unfinished games
                        if not status == "ready":
                            continue

                    fem_speakers = []
                    fem_dialogue = []

                    masc_speakers = []
                    masc_dialogue = []

                    div_speakers = []
                    div_dialogue = []

                    narrators = []
                    narrator_dialogue = []

                    for line in all_dialogues[gamevariant]:
                        speaker = line.get("character")
                        # if statement to match aliases; not needed because the parser took care of them
                        #if speaker in aliases:
                        #    speaker = aliases[speaker]
                        #    print(f"Replaced alias: {line.get('character')} with {speaker}")
                        # if character is female append line to fem_dialogue
                        if speaker in char_groups.get("female"):
                            fem_speakers.append(speaker)
                            fem_dialogue.append(line.get("line"))
                        # else if male append to male
                        elif speaker in char_groups.get("male"):
                            masc_speakers.append(speaker)
                            masc_dialogue.append(line.get("line"))
                        elif speaker in char_groups.get("neutral"):
                            if not str(speaker).lower() == "narrator":
                                div_speakers.append(speaker)
                                div_dialogue.append(line.get("line"))
                            else: #TODO damit auch irgendwas machen
                                narrators.append(speaker)
                                narrator_dialogue.append(line.get("line"))

                    female_pd = pd.DataFrame(list(zip(fem_speakers, fem_dialogue)),
                                      columns=['Name', 'Dialogue'])
                    male_pd = pd.DataFrame(list(zip(masc_speakers, masc_dialogue)),
                                             columns=['Name', 'Dialogue'])
                    div_pd = pd.DataFrame(list(zip(div_speakers, div_dialogue)),
                                           columns=['Name', 'Dialogue'])
                    gendered_dict = {'female': female_pd, 'male': male_pd, 'diverse': div_pd}

                    # add to according dict (df in dict in dict)
                    all_dialogues_fem[gamevariant] = female_pd
                    all_dialogues_male[gamevariant] = male_pd
                    all_dialogues_div[gamevariant] = div_pd

    # Create a structured DataFrame with all the dialogue
    df_dict = {}
    games_list = []
    for game, dialogues in all_dialogues.items():
        df_dict[game] = pd.DataFrame(dialogues)

    return df_dict, games_list, all_dialogues_fem, all_dialogues_male, all_dialogues_div, release_years


def extract_dialogue(dialogue_data, extracted_lines=None):
    '''
    Extract dialogue data from dialogue_data
    :param dialogue_data - contents from a json file
    :param extracted_lines - Boolean, introduced because of choice elements (recursive usage)
    :return: extracted_lines - dictionary, where entries are of format key=character, line=value
    '''
    # extract the dialogue lines according to readme provided by corpus website, recursively because of choice elements
    if extracted_lines is None:
        extracted_lines = []

    for entry in dialogue_data:
        if isinstance(entry, dict):
            for key, value in entry.items():
                if key == "CHOICE":
                    for sequence in value:
                        extract_dialogue(sequence, extracted_lines)
                elif key not in ("ACTION", "CHOICE"):
                    extracted_lines.append({"character": key, "line": value})
    return extracted_lines


def plot_stats(stats_fem, stats_male, stats_div):
    '''
    Plot statistics for all three gender groups.
    Input params are the output of process_folder() method.
    This method was used for an initial overview over the statistical data.
    :param stats_fem: dictionary, complete dialogue by gender (female)
    :param stats_male: dictionary, complete dialogue by gender (male)
    :param stats_div: dictionary, complete dialogue by gender (diverse)
    :return: None, plots the basic statistical data
    '''
    # Combine all three dataframes into one for easier plotting
    stats_fem['gender'] = 'Female'
    stats_male['gender'] = 'Male'
    stats_div['gender'] = 'Diverse'

    # Concatenate the three dataframes
    all_stats = pd.concat([stats_fem, stats_male, stats_div], axis=0)

    # 1. Bar plot for num_characters and length_dialogue_sum for each gamevariant
    gamevariants = all_stats['gamevariant'].unique()
    for gamevariant in gamevariants:
        plt.figure(figsize=(10, 6))
        subset = all_stats[all_stats['gamevariant'] == gamevariant]

        # Plot for num_characters
        plt.subplot(1, 2, 1)
        sns.barplot(x='gender', y='num_characters', data=subset, ci=None, palette='Set2')
        plt.title(f'Num Characters - {gamevariant}')
        plt.ylabel('Num Characters')

        # Plot for length_dialogue_sum
        plt.subplot(1, 2, 2)
        sns.barplot(x='gender', y='length_dialogue_sum', data=subset, ci=None, palette='Set2')
        plt.title(f'Length Dialogue Sum - {gamevariant}')
        plt.ylabel('Length Dialogue Sum')

        plt.tight_layout()
        plt.show()

    # 2. Line plot for num_characters and length_dialogue_sum for all gamevariants
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=all_stats, x='gamevariant', y='num_characters', hue='gender', marker='o')
    plt.title('Num Characters Across Gamevariants')
    plt.xlabel('Gamevariant')
    plt.ylabel('Num Characters')
    plt.legend(title='Gender')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=all_stats, x='gamevariant', y='length_dialogue_sum', hue='gender', marker='o')
    plt.title('Length Dialogue Sum Across Gamevariants')
    plt.xlabel('Gamevariant')
    plt.ylabel('Length Dialogue Sum')
    plt.legend(title='Gender')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def process_df_to_txt(dict_of_games, filename, separator="\n", rm_narrator=False):
    '''
    Processes a dataframe and returns all dialogue in txt format.
    :param: df - dataframe containing dialogue,
            filename - string to save txt file in,
            separator - string to determine if the text will be saved as one continuous string (" ")
                        or new line whenever someone new speaks (\n).
                        Default is new line.
            rm_narrator - Boolean, used to decide if diverse character 'Narrator' is kept in data.
                          Default is False.
    :return: texts - list of texts, all dialogue data
             gamevariants - list of game variants from which data was extracted
    '''
    texts = []
    gamevariants = []
    # read in each line of dataframe, append to string of dialogue
    for gamevariant, df_of_dialogue in dict_of_games.items():
        dialogue = ""
        if not rm_narrator:
            for text in df_of_dialogue['Dialogue']:
                dialogue = dialogue + text + separator

        else:
            print(type(df_of_dialogue), type(df_of_dialogue['Dialogue']))
            for index, row in df_of_dialogue.iterrows():
                if not "narrative" or "system" in str(row['Name']).lower():
                    text = str(row['Dialogue'])
                    dialogue = dialogue + text

        filename_var = filename + "_" + str(gamevariant).lower() + ".txt"
        with open(filename_var, 'w') as f:
            f.write(dialogue)
        texts.append(dialogue)
        gamevariants.append(gamevariant)

    return texts, gamevariants

def plot_dialogue_trends(absolute_counts_df, percentage_counts_df):
    '''
    Generates two plots: one for absolute dialogue counts and one for percentage
    of dialogue, broken down by gender over time.

    :param: absolute_counts_df  - DataFrame with absolute dialogue counts
                                           per year for each gender.
           percentage_counts_df  - DataFrame with percentage of dialogue
                                              per year for each gender.
    :return: None
    '''
    genders = ['male', 'female', 'diverse']
    colors = {'male': 'green', 'female': 'red', 'diverse': 'blue'}

    # Create a figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    fig.suptitle('Dialogue Trends Over Time by Gender', fontsize=16)

    # Plot Absolute Counts
    ax1 = axes[0]
    for gender in genders:
        if gender in absolute_counts_df.columns:
            ax1.plot(absolute_counts_df.index, absolute_counts_df[gender],
                     label=gender.capitalize(), color=colors[gender])
    ax1.set_title('Absolute Dialogue Counts by Year')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Number of Dialogues')
    ax1.legend()
    ax1.grid(True)
    ax1.set_xticks(absolute_counts_df.index) # Ensure all years are shown as ticks
    ax1.ticklabel_format(axis='y', style='plain') # Avoid scientific notation on y-axis
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right') # Rotate x-axis labels for better spacing

    # Plot Percentage Counts
    ax2 = axes[1]
    for gender in genders:
        if gender in percentage_counts_df.columns:
            ax2.plot(percentage_counts_df.index, percentage_counts_df[gender],
                     label=gender.capitalize(), color=colors[gender])
    ax2.set_title('Percentage of Dialogue by Year')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Percentage of Total Dialogue (%)')
    ax2.legend()
    ax2.grid(True)
    ax2.set_xticks(percentage_counts_df.index) # Ensure all years are shown as ticks
    ax2.ticklabel_format(axis='y', style='plain') # Avoid scientific notation on y-axis
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right') # Rotate x-axis labels for better spacing


    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to prevent title overlap
    plot_filename = os.path.join('dialogue-trends-over-time.png')
    plt.savefig(plot_filename)
    plt.close(fig)