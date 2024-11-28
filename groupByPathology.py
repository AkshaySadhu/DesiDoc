import pandas as pd
import os
import math

# Total number of rows in the dataset
total = 1025602

# Function to sample dataset and save all sampled data into one file
def sample_and_combine_dataset(input_file, output_file, sampling_factor=100000):
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv(input_file)

    # Initialize a list to store sampled data
    sampled_data = []

    print("Sampling dataset...")
    # Group by pathology and sample data
    for pathology, group in df.groupby('PATHOLOGY'):
        # Calculate the sample size for the pathology
        sample_size = math.ceil(len(group) / total * sampling_factor)
        print(f"Pathology: {pathology}, Rows: {len(group)}, Sample size: {sample_size}")

        # Sample the data (if sample size exceeds group size, use all rows)
        sampled_group = group.sample(n=min(sample_size, len(group)), random_state=42)

        # Append to the sampled data list
        sampled_data.append(sampled_group)

    # Combine all sampled data
    combined_data = pd.concat(sampled_data, ignore_index=True)

    # Save combined sampled data to a single file
    combined_data.to_csv(output_file, index=False)
    print(f"All sampled data saved to {output_file}")

# Main function
if __name__ == "__main__":
    # Specify input file and output file
    input_file = "C:/Users/urani/Documents/MSCS/CS6220 BDA/Project/HindiMedLLM/22687585/release_train_patients/release_train_patients"  # Replace with your dataset filename
    output_file = "C:/Users/urani/Documents/MSCS/CS6220 BDA/Project/HindiMedLLM/22687585/release_train_patients/sampled_train_combined_data_100k.csv"

    # Run the sampling function
    sample_and_combine_dataset(input_file, output_file)
