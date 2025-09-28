import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def emotion_analysis_per_game(female_csv, male_csv, diverse_csv):
    """
    Reads three CSV files, one for each gender (female, male, diverse),
    and generates box plots for each game variant.

    Args:
        female_csv (str): Path to the CSV file with female data.
        male_csv (str): Path to the CSV file with male data.
        diverse_csv (str): Path to the CSV file with diverse data.

    Author: @heidekrauttt, Google Gemini
    """
    try:
        # Read the three CSV files into DataFrames
        df_female = pd.read_csv(female_csv)
        df_male = pd.read_csv(male_csv)
        df_diverse = pd.read_csv(diverse_csv)

        # Concatenate the dataframes
        df_combined = pd.concat([df_female, df_male, df_diverse], ignore_index=True)

        # Create the output directory if it doesn't exist
        output_dir = "results/allgenders_base-emotions_utterancewise-nosurprise-fixlabel"
        os.makedirs(output_dir, exist_ok=True)

        # Identify the columns with highest count labels
        highest_count_cols = [
            col for col in df_combined.columns if col.startswith('highest_count_')
        ]

        # Get unique game variants
        game_variants = df_combined['gamevariant'].unique()

        # Define the custom color palette
        palette = {
            'female': 'red',
            'male': 'blue',
            'diverse': 'green'
        }

        # Loop through each game variant and create a plot
        for game in game_variants:
            # Filter the combined dataframe for the current game
            game_df = df_combined[df_combined['gamevariant'] == game].copy()

            # Melt the dataframe to long format for plotting
            df_melted = game_df.melt(
                id_vars=['gamevariant', 'gender'],
                value_vars=highest_count_cols,
                var_name='emotion_label',
                value_name='highest_count'
            )

            # Strip the 'highest_count_' prefix from the labels for better readability
            df_melted['emotion_label'] = df_melted['emotion_label'].str.replace('highest_count_', '', regex=False)

            # Create the box plot
            sns.barplot(
                data=df_melted,
                x='emotion_label',
                y='highest_count',
                hue='gender',
                palette=palette
            )
            plt.title(f'Frequency of Labels with Highest Score \n across Utterance for {game}')
            plt.xlabel('Emotion Label')
            plt.ylabel('Highest Score Count')
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Gender')
            plt.tight_layout()

            # Save the plot
            plot_filename = os.path.join(output_dir, f'{game}_boxplot.png')
            plt.savefig(plot_filename)
            plt.close()

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure all three CSV files exist in the specified directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def emotion_analysis_per_game_percentage(female_csv, male_csv, diverse_csv):
    """
    Reads three CSV files, one for each gender (female, male, diverse),
    converts absolute counts to percentages, and generates bar plots for
    each game variant.

    Args:
        female_csv (str): Path to the CSV file with female data.
        male_csv (str): Path to the CSV file with male data.
        diverse_csv (str): Path to the CSV file with diverse data.

    Author: @heidekrauttt, Google Gemini
    """
    try:
        # Read the three CSV files into DataFrames
        df_female = pd.read_csv(female_csv)
        df_male = pd.read_csv(male_csv)
        df_diverse = pd.read_csv(diverse_csv)

        # Create a list of the dataframes and a list of genders
        dfs = [df_female, df_male, df_diverse]

        # Identify the columns with highest count labels
        highest_count_cols = [
            col for col in df_female.columns if col.startswith('highest_count_')
        ]

        # Convert counts to percentages for each gender's dataframe
        for df in dfs:
            # Calculate the sum of all highest_count columns for each gamevariant/file
            df['total_counts'] = df[highest_count_cols].sum(axis=1)

            # Convert absolute counts to percentages
            for col in highest_count_cols:
                df[col] = (df[col] / df['total_counts']) * 100

        # Concatenate the processed dataframes
        df_combined = pd.concat(dfs, ignore_index=True)

        # Create the output directory if it doesn't exist
        output_dir = "results/allgenders_experimental-utterances-labels1"
        os.makedirs(output_dir, exist_ok=True)

        # Get unique game variants
        game_variants = df_combined['gamevariant'].unique()

        # Define the custom color palette
        palette = {
            'female': 'red',
            'male': 'blue',
            'diverse': 'green'
        }

        # Loop through each game variant and create a plot
        for game in game_variants:
            # Filter the combined dataframe for the current game
            game_df = df_combined[df_combined['gamevariant'] == game].copy()

            # Melt the dataframe to long format for plotting
            df_melted = game_df.melt(
                id_vars=['gamevariant', 'gender'],
                value_vars=highest_count_cols,
                var_name='emotion_label',
                value_name='percentage'
            )

            # Strip the 'highest_count_' prefix from the labels for better readability
            df_melted['emotion_label'] = df_melted['emotion_label'].str.replace('highest_count_', '', regex=False)

            # Create the bar plot
            plt.figure(figsize=(12, 8))
            sns.barplot(
                data=df_melted,
                x='emotion_label',
                y='percentage',
                hue='gender',
                palette=palette
            )
            plt.title(f'Percentage of Labels with Highest Score across Utterance for {game}')
            plt.xlabel('Emotion Label')
            plt.ylabel('Percentage')
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Gender')
            plt.tight_layout()

            # Save the plot
            plot_filename = os.path.join(output_dir, f'{game}_percentage_barplot.png')
            plt.savefig(plot_filename)
            plt.close()

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure all three CSV files exist in the specified directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def emotion_analysis_mean_percentage(output_path, female_csv, male_csv, diverse_csv):
    """
    Reads three CSV files, one for each gender (female, male, diverse),
    converts absolute counts to percentages, and generates a single bar plot
    showing the mean percentage of labels across all games.

    Args:
        female_csv (str): Path to the CSV file with female data.
        male_csv (str): Path to the CSV file with male data.
        diverse_csv (str): Path to the CSV file with diverse data.

    Author: @heidekrauttt, Google Gemini
    """
    try:
        # Read the three CSV files into DataFrames
        df_female = pd.read_csv(female_csv)
        df_male = pd.read_csv(male_csv)
        df_diverse = pd.read_csv(diverse_csv)

        # Create a list of the dataframes and a list of genders
        dfs = [df_female, df_male, df_diverse]
        genders = ['female', 'male', 'diverse']

        # Add a 'gender' column to each dataframe for later grouping
        for i, df in enumerate(dfs):
            df['gender'] = genders[i]

        # Identify the columns with highest count labels
        # Assuming all dataframes have the same structure
        highest_count_cols = [
            col for col in df_female.columns if col.startswith('highest_count_')
        ]

        # Convert counts to percentages for each gender's dataframe
        for df in dfs:
            # Calculate the sum of all highest_count columns for each row
            df['total_counts'] = df[highest_count_cols].sum(axis=1)

            # Convert absolute counts to percentages
            for col in highest_count_cols:
                df[col] = (df[col] / df['total_counts']) * 100

        # Concatenate the processed dataframes
        df_combined = pd.concat(dfs, ignore_index=True)

        # Group by gender and calculate the mean percentage for each emotion label
        df_mean = df_combined.groupby('gender')[highest_count_cols].mean().reset_index()

        # Melt the dataframe to long format for plotting
        df_melted = df_mean.melt(
            id_vars=['gender'],
            value_vars=highest_count_cols,
            var_name='emotion_label',
            value_name='percentage'
        )

        # Strip the 'highest_count_' prefix from the labels for better readability
        df_melted['emotion_label'] = df_melted['emotion_label'].str.replace('highest_count_', '', regex=False)

        # Define the custom color palette
        palette = {
            'female': 'red',
            'male': 'blue',
            'diverse': 'green'
        }

        # Create the output directory if it doesn't exist
        output_dir = output_path
        os.makedirs(output_dir, exist_ok=True)

        # Create the single bar plot
        plt.figure(figsize=(12, 8))
        sns.barplot(
            data=df_melted,
            x='emotion_label',
            y='percentage',
            hue='gender',
            palette=palette
        )
        plt.title('Mean Percentage of Labels with Highest Score across all Games')
        plt.xlabel('Emotion Label')
        plt.ylabel('Mean Percentage')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Gender')
        plt.tight_layout()

        # Save the plot
        plot_filename = os.path.join(output_dir, 'all_games_mean_percentage_barplot.png')
        plt.savefig(plot_filename)
        plt.close()

    except FileNotFoundError as e:
        print(f"Error: One of the specified CSV files was not found. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def plot_sentiments(file_female: str, file_male: str, file_diverse: str, output_dir: str = 'sentiment_plots-fmd'):
    """
    Plots the mean negative, positive, and neutral sentiment values for each game variant
    from three CSV files (female, male, and diverse sentiments) using grouped bar charts.

    Each plot will show bars for female (red), male (green), and diverse (blue) sentiments,
    side-by-side for negative, neutral, and positive categories.
    If the diverse sentiment file is empty or not found, only female and male sentiments will be plotted.

    Args:
        file_female (str): Path to the CSV file containing female sentiment analysis results.
        file_male (str): Path to the CSV file containing male sentiment analysis results.
        file_diverse (str): Path to the CSV file containing diverse sentiment analysis results.
                            If this file is empty or not found, it will be skipped.
        output_dir (str): Directory to save the generated plots. Defaults to 'sentiment_plots'.

    Author: @heidekrauttt, Google Gemini
    """
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
                bars_male = ax.bar(x, male_sentiments, width, label='Male', color='blue', alpha=0.8)
                bars_diverse = ax.bar(x + width, diverse_sentiments, width, label='Diverse', color='green', alpha=0.8)
                all_sentiments.extend(female_sentiments + male_sentiments + diverse_sentiments)
            else:
                width = 0.35 # The width of the bars for two groups
                bars_female = ax.bar(x - width/2, female_sentiments, width, label='Female', color='red', alpha=0.8)
                bars_male = ax.bar(x + width/2, male_sentiments, width, label='Male', color='blue', alpha=0.8)
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


def emotion_analysis_mean_frequency_percentages(output_dir, female_csv, male_csv, diverse_csv):
    """
    Reads three CSV files, one for each gender (female, male, diverse),
    converts absolute counts to percentages, and generates a single bar plot
    showing the mean frequency of Labels with highest score across all Games.

    Args:
        female_csv (str): Path to the CSV file with female data.
        male_csv (str): Path to the CSV file with male data.
        diverse_csv (str): Path to the CSV file with diverse data.

    Author: @heidekrauttt, Google Gemini
    """
    try:
        # Read the three CSV files into DataFrames
        df_female = pd.read_csv(female_csv)
        df_male = pd.read_csv(male_csv)
        df_diverse = pd.read_csv(diverse_csv)

        # Create a list of the dataframes and a list of genders
        dfs = [df_female, df_male, df_diverse]
        genders = ['female', 'male', 'diverse']

        # Add a 'gender' column to each dataframe for later grouping
        for i, df in enumerate(dfs):
            df['gender'] = genders[i]

        # Identify the columns for highest counts and mean scores
        highest_count_cols = [
            col for col in df_female.columns if col.startswith('highest_count_')
        ]
        mean_cols = [
            col for col in df_female.columns if col.startswith('mean_')
        ]

        # Concatenate the processed dataframes
        df_combined = pd.concat(dfs, ignore_index=True)

        # Create the output directory if it doesn't exist
        output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # --- Plot 1: Mean Percentage of Highest Count Labels ---
        # Convert counts to percentages for each gender's dataframe
        for df in dfs:
            # Calculate the sum of all highest_count columns for each row
            df['total_counts'] = df[highest_count_cols].sum(axis=1)

            # Convert absolute counts to percentages
            for col in highest_count_cols:
                df[col] = (df[col] / df['total_counts']) * 100

        # Group by gender and calculate the mean percentage for each emotion label
        df_mean = df_combined.groupby('gender')[highest_count_cols].mean().reset_index()

        # Melt the dataframe to long format for plotting
        df_melted = df_mean.melt(
            id_vars=['gender'],
            value_vars=highest_count_cols,
            var_name='emotion_label',
            value_name='percentage'
        )

        # Strip the 'highest_count_' prefix from the labels for better readability
        df_melted['emotion_label'] = df_melted['emotion_label'].str.replace('highest_count_', '', regex=False)

        # Define the custom color palette
        palette = {
            'female': 'red',
            'male': 'blue',
            'diverse': 'green'
        }

        # Create the bar plot
        plt.figure(figsize=(12, 8))
        sns.barplot(
            data=df_melted,
            x='emotion_label',
            y='percentage',
            hue='gender',
            palette=palette
        )
        plt.title('Mean Frequency of Labels with Highest Score across all Games')
        plt.xlabel('Emotion Label')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Gender')
        plt.tight_layout()

        # Save the plot
        plot_filename = os.path.join(output_dir, 'all_games_mean_percentage_barplot.png')
        plt.savefig(plot_filename)
        plt.close()

        # --- Plot 2: Mean Scores of All Labels ---
        # Group by gender and calculate the mean score for each 'mean_' label
        df_mean_scores = df_combined.groupby('gender')[mean_cols].mean().reset_index()

        # Melt the dataframe to long format for plotting
        df_mean_scores_melted = df_mean_scores.melt(
            id_vars=['gender'],
            value_vars=mean_cols,
            var_name='emotion_label',
            value_name='mean_score'
        )

        # Strip the 'mean_' prefix from the labels for better readability
        df_mean_scores_melted['emotion_label'] = df_mean_scores_melted['emotion_label'].str.replace('mean_', '',
                                                                                                    regex=False)

        # Create the second bar plot
        plt.figure(figsize=(12, 8))
        sns.barplot(
            data=df_mean_scores_melted,
            x='emotion_label',
            y='mean_score',
            hue='gender',
            palette=palette
        )
        plt.title('Mean Scores of All Labels across all Games')
        plt.xlabel('Emotion Label')
        plt.ylabel('Mean Score')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Gender')
        plt.tight_layout()

        # Save the second plot
        plot_filename_mean = os.path.join(output_dir, 'all_games_mean_scores_barplot.png')
        plt.savefig(plot_filename_mean)
        plt.close()


    except FileNotFoundError as e:
        print(f"Error: One of the specified CSV files was not found. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":

    # Example usage of the plotting functions in this file.

    plot_sentiments("sentiment_analysis_results-female.csv", "sentiment_analysis_results-male.csv",
                    "sentiment_analysis_results-diverse.csv", output_dir="sentiment_plots/fmd-sentiments")

    emotion_analysis_per_game("/emotion_analysis-utterancewise-nosurprise-resultsmale.csv",
                              "/emotion_analysis-utterancewise-nosurprise-resultsfemale.csv",
                              "/emotion_analysis-utterancewise-nosurprise-resultsdiverse.csv")
    emotion_analysis_per_game_percentage("/emotion_analysis-utterancewise-experimentalmale.csv",
                                         "/emotion_analysis-utterancewise-experimentalfemale.csv",
                                         "/emotion_analysis-utterancewise-experimentaldiverse.csv")

    emotion_analysis_mean_percentage("/results/allgenders_base-emotions_utterancewise-nosurprise",
                                     "/emotion_analysis_results-female.csv",
                                     "/emotion_analysis_results-male.csv",
                                     "/emotion_analysis_results-diverse.csv")

    emotion_analysis_mean_frequency_percentages("/results/allgenders_experimental-utterances-labels1",
                                      "/emotion_analysis-utterancewise-experimentalfemale.csv",
                                      "/emotion_analysis-utterancewise-experimentalmale.csv",
                                      "/emotion_analysis-utterancewise-experimentaldiverse.csv")
    emotion_analysis_mean_frequency_percentages("/results/allgenders_experimental-utterances-labels2",
                                      "/emotion_analysis-utterancewise-experimental-labels2female.csv",
                                      "/emotion_analysis-utterancewise-experimental-labels2male.csv",
                                      "/0824-emotion_analysis-utterancewise-experimental-labels2diverse.csv")
    emotion_analysis_mean_frequency_percentages("/results/allgenders_base-emotions_utterancewise-nosurprise",
                                      "/emotion_analysis_results-female.csv",
                                      "/emotion_analysis_results-male.csv",
                                      "/emotion_analysis_results-diverse.csv")


