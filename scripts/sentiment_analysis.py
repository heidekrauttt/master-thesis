import nltk
import transformers
from transformers import pipeline
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

def emotion_analysis(gender, input_folder="../data/txt_files/", output_folder="results/base-emotion_analysis", csv_filename="sentiment_analysis_results_", plot=False, splitby="speaker"):
    '''
    analyze emotion of sentences or utterances, using DeBERTaV3 transformer model and zero shot classification.
    Used labels: base emotions by Paul Ekman, either with or without label Suprise (manual selection)
    :param gender: specifies which gender to analyze
    :param input_folder: path to textual data (.txt files)
    :param output_folder: output path where results are saved
    :param csv_filename: filename to save results as .csv file to
    :param plot: Boolean, if results are plotted
    :param splitby: split dialogue by Speaker or linewise. Default: speaker
    :return: None
    '''
    # model taken from https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
    classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

    # ---------- emotions -----------------
    # base emotions as candidate labels, taken from Paul Ekman, including Surprise
    #candidate_labels = ["Neutral", "Joy", "Sadness", "Fear", "Anger", "Surprise", "Disgust"]
    # base emotions as candidate labels, excluding Surprise
    candidate_labels = ["Neutral", "Joy", "Sadness", "Fear", "Anger", "Disgust"]

    # Initialize an empty pandas DataFrame to store all results
    # add columns dynamically based on the candidate_labels and other relevant info
    all_results_df = pd.DataFrame()

    for file in os.listdir(input_folder):
        start = time.time()
        # regex to achieve gamevariant name
        gamevariant = str(file).replace(".txt", "").split("_")[-1]

        if file.endswith(".txt"):
            all_outputs = [] # Reset for each file
            with open(os.path.join(input_folder, file), "r") as f:
                text = f.read()
                # make sure text exists and the file is not empty
                if not text:
                    continue
                if splitby == "speaker":
                    text = text.split("\n") # splits by speaker
                elif splitby == "sentence":
                    text = nltk.sent_tokenize(text) # splits by sentence
                for sentence in text:
                    if sentence != "":
                        output = classifier(sentence, candidate_labels, multi_label=False)
                        all_outputs.append(output)

                end = time.time()
                print(f"{file} sentiment analyzed in {end-start} seconds")

                # --- Data Processing ---
                highest_score_counts = {label: 0 for label in candidate_labels}
                all_label_scores = {label: [] for label in candidate_labels}

                if all_outputs:
                    for output in all_outputs:
                        labels = output['labels']
                        scores = output['scores']

                        max_score_index = np.argmax(scores)
                        highest_label = labels[max_score_index]
                        highest_score_counts[highest_label] += 1

                        for i, label in enumerate(labels):
                            all_label_scores[label].append(scores[i])

                mean_scores = {}
                for label, scores_list in all_label_scores.items():
                    if scores_list:
                        mean_scores[label] = np.mean(scores_list)
                    else:
                        mean_scores[label] = 0

                # Create a temporary DataFrame for the current file's results
                # This DataFrame will contain the mean scores and highest score counts for the current file
                current_file_data = {'file': file, 'gamevariant': gamevariant, 'gender': gender}

                # Add mean scores for each emotion
                for label in candidate_labels:
                    current_file_data[f'mean_{label}'] = mean_scores.get(label, 0)

                # Add highest score counts for each emotion
                for label in candidate_labels:
                    current_file_data[f'highest_count_{label}'] = highest_score_counts.get(label, 0)

                # Convert to a pandas Series and then to a DataFrame to ensure it's row-wise
                current_file_df = pd.DataFrame([current_file_data])

                # Concatenate with the main DataFrame
                all_results_df = pd.concat([all_results_df, current_file_df], ignore_index=True)

                # --- Plotting, optional step (selection through Boolean plot) ---
                if plot:
                    output_base_dir = output_folder
                    output_dir = os.path.join(output_base_dir, gender, gamevariant)
                    os.makedirs(output_dir, exist_ok=True)

                    # Plot 1: Frequency of labels with the highest score
                    labels_highest = list(highest_score_counts.keys())
                    counts_highest = list(highest_score_counts.values())

                    plt.figure(figsize=(10, 6))
                    plt.bar(labels_highest, counts_highest, color='skyblue')
                    plt.xlabel('Label')
                    plt.ylabel('Frequency (Times Highest Score)')
                    plt.title('Frequency of Labels with Highest Score Across Sentences for ' + gamevariant)
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'highest_score_frequency_plot.png'))
                    plt.close()

                    # Plot 2: Mean score for each label
                    labels_mean = list(mean_scores.keys())
                    scores_mean = list(mean_scores.values())

                    plt.figure(figsize=(10, 6))
                    plt.bar(labels_mean, scores_mean, color='lightcoral')
                    plt.xlabel('Label')
                    plt.ylabel('Mean Score')
                    plt.title('Mean Score for Each Label Across All Sentences for ' + gamevariant)
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'mean_score_plot.png'))
                    plt.close()

    # Save the DataFrame to a CSV file
    csv_path = csv_filename + str(gender) + ".csv"
    all_results_df.to_csv(csv_path, index=False)


def emotion_analysis_experimental(gender, input_folder="../data/txt_files/", output_folder="results/base-emotion_analysis-experimental", csv_filename="sentiment_analysis_results_", plot=False, splitby="speaker"):
    '''
        analyze emotion of sentences or utterances, using DeBERTaV3 transformer model and zero shot classification.
        Used labels: experimental labels from master thesis, based on masculinity studies.
                     Manual choice of two sets of labels.
        :param gender: specifies which gender to analyze
        :param input_folder: path to textual data (.txt files)
        :param output_folder: output path where results are saved
        :param csv_filename: filename to save results as .csv file to
        :param plot: Boolean, if results are plotted
        :param splitby: split dialogue by Speaker or linewise. Default: speaker
        :return: None
        '''
    # model taken from https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
    classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", framework="pt")

    # ---------- Experimental labels -----------------
    # Experimental Label set 1
    #candidate_labels = ["Neutral", "Helpless", "Confident", "Angry", "Sad", "Dominant", "Submissive"]
    # Experimental Label set 2
    candidate_labels = ["Neutral", "Resistant", "Receptive", "Powerful", "Gentle", "Caring", "Stoic", "Dominant", "Submissive"]

    # Initialize an empty pandas DataFrame to store all results
    # We'll add columns dynamically based on the candidate_labels and other relevant info
    all_results_df = pd.DataFrame()

    for file in os.listdir(input_folder):
        start = time.time()
        gamevariant = str(file).replace(".txt", "").split("_")[-1]

        if file.endswith(".txt"):
            all_outputs = [] # Reset for each file
            with open(os.path.join(input_folder, file), "r") as f:
                text = f.read()
                # make sure text exists and the file is not empty
                if not text:
                    continue
                if splitby == "speaker":
                    text = text.split("\n") # splits by speaker
                elif splitby == "sentence":
                    text = nltk.sent_tokenize(text) # splits by sentence
                for sentence in text:
                    if sentence != "":
                        output = classifier(sentence, candidate_labels, multi_label=False)
                        all_outputs.append(output)

                end = time.time()
                print(f"{file} sentiment analyzed in {end-start} seconds")

                # --- Data Processing ---
                highest_score_counts = {label: 0 for label in candidate_labels}
                all_label_scores = {label: [] for label in candidate_labels}

                if all_outputs:
                    for output in all_outputs:
                        labels = output['labels']
                        scores = output['scores']

                        max_score_index = np.argmax(scores)
                        highest_label = labels[max_score_index]
                        highest_score_counts[highest_label] += 1

                        for i, label in enumerate(labels):
                            all_label_scores[label].append(scores[i])

                mean_scores = {}
                for label, scores_list in all_label_scores.items():
                    if scores_list:
                        mean_scores[label] = np.mean(scores_list)
                    else:
                        mean_scores[label] = 0

                # Create a temporary DataFrame for the current file's results
                # This DataFrame will contain the mean scores and highest score counts for the current file
                current_file_data = {'file': file, 'gamevariant': gamevariant, 'gender': gender}

                # Add mean scores for each emotion
                for label in candidate_labels:
                    current_file_data[f'mean_{label}'] = mean_scores.get(label, 0)

                # Add highest score counts for each emotion
                for label in candidate_labels:
                    current_file_data[f'highest_count_{label}'] = highest_score_counts.get(label, 0)

                # Convert to a pandas Series and then to a DataFrame to ensure it's row-wise
                current_file_df = pd.DataFrame([current_file_data])

                # Concatenate with the main DataFrame
                all_results_df = pd.concat([all_results_df, current_file_df], ignore_index=True)


                if plot:
                    output_base_dir = output_folder
                    output_dir = os.path.join(output_base_dir, gender, gamevariant)
                    os.makedirs(output_dir, exist_ok=True)

                    # Plot 1: Frequency of labels with the highest score
                    labels_highest = list(highest_score_counts.keys())
                    counts_highest = list(highest_score_counts.values())

                    plt.figure(figsize=(10, 6))
                    plt.bar(labels_highest, counts_highest, color='skyblue')
                    plt.xlabel('Label')
                    plt.ylabel('Frequency (Times Highest Score)')
                    plt.title('Frequency of Labels with Highest Score Across Sentences for ' + gamevariant)
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'highest_score_frequency_plot.png'))
                    plt.close()

                    # Plot 2: Mean score for each label
                    labels_mean = list(mean_scores.keys())
                    scores_mean = list(mean_scores.values())

                    plt.figure(figsize=(10, 6))
                    plt.bar(labels_mean, scores_mean, color='lightcoral')
                    plt.xlabel('Label')
                    plt.ylabel('Mean Score')
                    plt.title('Mean Score for Each Label Across All Sentences for ' + gamevariant)
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'mean_score_plot.png'))
                    plt.close()

    # Save the DataFrame to a CSV file
    csv_path = csv_filename + str(gender) + ".csv"
    all_results_df.to_csv(csv_path, index=False)


def sentiment_analysis(gender, input_folder="../data/txt_files/", output_folder="results/sentiment_analysis", csv_filename="sentiment_analysis_results_", plot=False, splitby="sentence"):
    '''
    analyze sentiment of sentences or utterances, using DeBERTaV3 transformer model and zero shot classification.
    :param gender: specifies which gender to analyze
    :param input_folder: path to textual data (.txt files)
    :param output_folder: output path where results are saved
    :param csv_filename: filename to save results as .csv file to
    :param plot: Boolean, if results are plotted
    :param splitby: split dialogue by Speaker or linewise. Default: speaker
    :return: None
    '''
    # ---------- sentiments -----------------
    classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

    # sentiment analysis with classic labels
    candidate_labels = ["negative", "positive", "neutral"]
    # Initialize an empty pandas DataFrame to store all results
    # add columns dynamically based on the candidate_labels and other relevant info
    all_results_df = pd.DataFrame()

    for file in os.listdir(input_folder):
        start = time.time()
        # get gamevariant using a regex
        gamevariant = str(file).replace(".txt", "").split("_")[-1]

        if file.endswith(".txt"):
            all_outputs = [] # Reset for each file
            with open(os.path.join(input_folder, file), "r") as f:
                text = f.read()
                if splitby == "speaker":
                    text = text.split("\n")  # splits by speaker
                elif splitby == "sentence":
                    text = nltk.sent_tokenize(text)  # splits by sentence
                for sentence in text:
                    # actual classification with transformer model
                    output = classifier(sentence, candidate_labels, multi_label=False)
                    all_outputs.append(output)

                end = time.time()
                print(f"{file} sentiment analyzed in {end-start} seconds")

                # --- Data Processing ---
                highest_score_counts = {label: 0 for label in candidate_labels}
                all_label_scores = {label: [] for label in candidate_labels}

                if all_outputs:
                    for output in all_outputs:
                        labels = output['labels']
                        scores = output['scores']

                        max_score_index = np.argmax(scores)
                        highest_label = labels[max_score_index]
                        highest_score_counts[highest_label] += 1

                        for i, label in enumerate(labels):
                            all_label_scores[label].append(scores[i])

                mean_scores = {}
                for label, scores_list in all_label_scores.items():
                    if scores_list:
                        mean_scores[label] = np.mean(scores_list)
                    else:
                        mean_scores[label] = 0

                # Create a temporary DataFrame for the current file's results
                # This DataFrame will contain the mean scores and highest score counts for the current file
                current_file_data = {'file': file, 'gamevariant': gamevariant, 'gender': gender}

                # Add mean scores for each emotion
                for label in candidate_labels:
                    current_file_data[f'mean_{label}'] = mean_scores.get(label, 0)

                # Add highest score counts for each emotion
                for label in candidate_labels:
                    current_file_data[f'highest_count_{label}'] = highest_score_counts.get(label, 0)

                # Convert to a pandas Series and then to a DataFrame to ensure it's row-wise
                current_file_df = pd.DataFrame([current_file_data])

                # Concatenate with the main DataFrame
                all_results_df = pd.concat([all_results_df, current_file_df], ignore_index=True)


                if plot:
                    output_base_dir = output_folder
                    output_dir = os.path.join(output_base_dir, gender, gamevariant)
                    os.makedirs(output_dir, exist_ok=True)

                    # Plot 1: Frequency of labels with the highest score
                    labels_highest = list(highest_score_counts.keys())
                    counts_highest = list(highest_score_counts.values())

                    plt.figure(figsize=(10, 6))
                    plt.bar(labels_highest, counts_highest, color='skyblue')
                    plt.xlabel('Label')
                    plt.ylabel('Frequency (Times Highest Score)')
                    plt.title('Frequency of Labels with Highest Score Across Sentences for ' + gamevariant)
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'highest_score_frequency_plot.png'))
                    plt.close()

                    # Plot 2: Mean score for each label
                    labels_mean = list(mean_scores.keys())
                    scores_mean = list(mean_scores.values())

                    plt.figure(figsize=(10, 6))
                    plt.bar(labels_mean, scores_mean, color='lightcoral')
                    plt.xlabel('Label')
                    plt.ylabel('Mean Score')
                    plt.title('Mean Score for Each Label Across All Sentences for ' + gamevariant)
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'mean_score_plot.png'))
                    plt.close()

    # save to csv
    csv_path = csv_filename + str(gender) +".csv"
    all_results_df.to_csv(csv_path, index=False)


def plot_two_sentiments(file_female: str, file_male: str, output_dir: str = 'sentiment_plots'):
    '''
    Plots the mean negative, positive, and neutral sentiment values for each game variant
    from two CSV files (one for female sentiments, one for male sentiments) using grouped bar charts.

    Each plot will show 6 bars: mean negative, mean neutral, and mean positive for female
    (in red), and the corresponding values for male (in green), side-by-side.

    :param:
        file_female (str): Path to the CSV file containing female sentiment analysis results.
        file_male (str): Path to the CSV file containing male sentiment analysis results.
        output_dir (str): Directory to save the generated plots. Defaults to 'sentiment_plots'.
    '''
    try:
        # Load the sentiment analysis results from the CSV files
        df_female = pd.read_csv(file_female)
        df_male = pd.read_csv(file_male)

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Group by 'game_variant' and calculate the mean of 'mean_negative', 'mean_positive',
        # and 'mean_neutral' for both dataframes.
        # We assume these columns exist in the input CSVs.
        sentiment_cols = ['mean_negative', 'mean_positive', 'mean_neutral']
        df_female_grouped = df_female.groupby('gamevariant')[sentiment_cols].mean().reset_index()
        df_male_grouped = df_male.groupby('gamevariant')[sentiment_cols].mean().reset_index()

        # Merge the grouped dataframes on 'game_variant' to easily compare
        merged_df = pd.merge(df_female_grouped, df_male_grouped, on='gamevariant', suffixes=('_female', '_male'))

        # Iterate through each game variant and create a plot
        for index, row in merged_df.iterrows():
            game_variant = row['gamevariant']

            # Extract sentiment values for female and male
            female_sentiments = [
                row['mean_negative_female'],
                row['mean_neutral_female'],
                row['mean_positive_female']
            ]
            male_sentiments = [
                row['mean_negative_male'],
                row['mean_neutral_male'],
                row['mean_positive_male']
            ]

            # Create a new figure for each game variant
            fig, ax = plt.subplots(figsize=(10, 7)) # Adjust figure size for better readability

            # Data for plotting
            sentiment_types = ['Negative', 'Neutral', 'Positive']
            x = np.arange(len(sentiment_types)) # The label locations
            width = 0.35 # The width of the bars

            # Create grouped bar plots
            bars_female = ax.bar(x - width/2, female_sentiments, width, label='Female', color='red', alpha=0.8)
            bars_male = ax.bar(x + width/2, male_sentiments, width, label='Male', color='green', alpha=0.8)

            # Add labels and title
            ax.set_ylabel('Mean Sentiment Score', fontsize=12)
            ax.set_title(f'Mean Sentiment Scores for Game Variant: {game_variant}', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(sentiment_types, fontsize=10)
            ax.set_ylim(min(0, min(min(female_sentiments), min(male_sentiments)) - 0.1),
                        max(1, max(max(female_sentiments), max(male_sentiments)) + 0.1)) # Adjust Y-axis limits

            # Add value labels on top of the bars
            def add_value_labels(bars):
                for bar in bars:
                    yval = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 3), ha='center', va='bottom', fontsize=9)

            add_value_labels(bars_female)
            add_value_labels(bars_male)

            # Add a legend
            ax.legend(title='Gender', fontsize=10)

            # Add a grid for better readability
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            # Improve layout and save the plot
            plt.tight_layout()
            plot_filename = os.path.join(output_dir, f'sentiment_plot_{game_variant.replace(" ", "_").replace("/", "_")}.png')
            plt.savefig(plot_filename)
            plt.close(fig) # Close the figure to free memory

        print(f"Plots saved successfully in the '{output_dir}' directory.")

    except FileNotFoundError:
        print("Error: One or both of the CSV files were not found. Please check the paths.")
    except KeyError as e:
        print(f"Error: Missing expected column in CSV file. Please ensure 'game_variant', 'mean_negative', 'mean_positive', and 'mean_neutral' columns exist. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def plot_sentiments(file_female: str, file_male: str, file_diverse: str, output_dir: str = 'sentiment_plots-fmd'):
    '''
    Plots the mean negative, positive, and neutral sentiment values for each game variant
    from three CSV files (female, male, and diverse sentiments) using grouped bar charts.

    Each plot will show bars for female (red), male (green), and diverse (blue) sentiments,
    side-by-side for negative, neutral, and positive categories.
    If the diverse sentiment file is empty or not found, only female and male sentiments will be plotted.

    :param:
        file_female (str): Path to the CSV file containing female sentiment analysis results.
        file_male (str): Path to the CSV file containing male sentiment analysis results.
        file_diverse (str): Path to the CSV file containing diverse sentiment analysis results.
                            If this file is empty or not found, it will be skipped.
        output_dir (str): Directory to save the generated plots. Defaults to 'sentiment_plots'.
    '''
    try:
        # Load the sentiment analysis results from the CSV files
        df_female = pd.read_csv(file_female)
        df_male = pd.read_csv(file_male)

        df_diverse = None # Initialize df_diverse to None
        try:
            df_diverse_temp = pd.read_csv(file_diverse)
            # Check if the diverse dataframe is not empty
            if not df_diverse_temp.empty:
                df_diverse = df_diverse_temp
            else:
                print(f"Warning: The diverse sentiment file '{file_diverse}' is empty. Plotting only female and male sentiments.")
        except FileNotFoundError:
            print(f"Warning: The diverse sentiment file '{file_diverse}' was not found. Plotting only female and male sentiments.")
        except Exception as e:
            print(f"Warning: An error occurred while loading diverse sentiment file '{file_diverse}': {e}. Plotting only female and male sentiments.")


        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Group by 'game_variant' and calculate the mean of 'mean_negative', 'mean_positive',
        # and 'mean_neutral' for all dataframes.
        # We assume these columns exist in the input CSVs.
        sentiment_cols = ['mean_negative', 'mean_positive', 'mean_neutral']
        df_female_grouped = df_female.groupby('gamevariant')[sentiment_cols].mean().reset_index()
        df_male_grouped = df_male.groupby('gamevariant')[sentiment_cols].mean().reset_index()

        # Merge the grouped dataframes on 'gamevariant' to easily compare
        merged_df = pd.merge(df_female_grouped, df_male_grouped, on='gamevariant', suffixes=('_female', '_male'))

        if df_diverse is not None:
            df_diverse_grouped = df_diverse.groupby('gamevariant')[sentiment_cols].mean().reset_index()
            # Explicitly rename columns in df_diverse_grouped to ensure distinct names before merging
            df_diverse_grouped = df_diverse_grouped.rename(columns={
                'mean_negative': 'mean_negative_diverse',
                'mean_positive': 'mean_positive_diverse',
                'mean_neutral': 'mean_neutral_diverse'
            })
            # Merge diverse data. Suffixes parameter is no longer needed here as columns are pre-renamed.
            merged_df = pd.merge(merged_df, df_diverse_grouped, on='gamevariant')

        # Iterate through each game variant and create a plot
        for index, row in merged_df.iterrows():
            game_variant = row['gamevariant']

            # Extract sentiment values for female and male
            female_sentiments = [
                row['mean_negative_female'],
                row['mean_neutral_female'],
                row['mean_positive_female']
            ]
            male_sentiments = [
                row['mean_negative_male'],
                row['mean_neutral_male'],
                row['mean_positive_male']
            ]

            diverse_sentiments = None
            # Check if diverse data exists for this specific game variant in the merged_df
            if df_diverse is not None and 'mean_negative_diverse' in row:
                diverse_sentiments = [
                    row['mean_negative_diverse'],
                    row['mean_neutral_diverse'],
                    row['mean_positive_diverse']
                ]

            # Create a new figure for each game variant
            fig, ax = plt.subplots(figsize=(12, 7)) # Adjust figure size for better readability

            # Data for plotting
            sentiment_types = ['Negative', 'Neutral', 'Positive']
            x = np.arange(len(sentiment_types)) # The label locations

            all_sentiments = []

            # Determine bar width and positions based on whether diverse data is present
            if diverse_sentiments is not None:
                width = 0.25 # The width of the bars for three groups
                bars_female = ax.bar(x - width, female_sentiments, width, label='Female', color='red', alpha=0.8)
                bars_male = ax.bar(x, male_sentiments, width, label='Male', color='green', alpha=0.8)
                bars_diverse = ax.bar(x + width, diverse_sentiments, width, label='Diverse', color='blue', alpha=0.8)
                all_sentiments.extend(female_sentiments + male_sentiments + diverse_sentiments)
            else:
                width = 0.35 # The width of the bars for two groups
                bars_female = ax.bar(x - width/2, female_sentiments, width, label='Female', color='red', alpha=0.8)
                bars_male = ax.bar(x + width/2, male_sentiments, width, label='Male', color='green', alpha=0.8)
                all_sentiments.extend(female_sentiments + male_sentiments)


            # Add labels and title
            ax.set_ylabel('Mean Sentiment Score', fontsize=12)
            ax.set_title(f'Mean Sentiment Scores for Game Variant: {game_variant}', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(sentiment_types, fontsize=10)

            # Adjust Y-axis limits based on all visible sentiments
            min_y = min(0, min(all_sentiments) - 0.1) if all_sentiments else 0
            max_y = max(1, max(all_sentiments) + 0.1) if all_sentiments else 1
            ax.set_ylim(min_y, max_y)

            # Add value labels on top of the bars
            def add_value_labels(bars):
                for bar in bars:
                    yval = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 3), ha='center', va='bottom', fontsize=9)

            add_value_labels(bars_female)
            add_value_labels(bars_male)
            if diverse_sentiments is not None:
                add_value_labels(bars_diverse)

            # Add a legend
            ax.legend(title='Gender', fontsize=10)

            # Add a grid for better readability
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            # Improve layout and save the plot
            plt.tight_layout()
            plot_filename = os.path.join(output_dir, f'sentiment_plot_{game_variant.replace(" ", "_").replace("/", "_")}.png')
            plt.savefig(plot_filename)
            plt.close(fig) # Close the figure to free memory

        print(f"Plots saved successfully in the '{output_dir}' directory.")

    except FileNotFoundError:
        print("Error: One or more of the required CSV files (female/male/diverse) were not found. Please check the paths.")
    except KeyError as e:
        print(f"Error: Missing expected column in CSV file. Please ensure 'gamevariant', 'mean_negative', 'mean_positive', and 'mean_neutral' columns exist. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")



