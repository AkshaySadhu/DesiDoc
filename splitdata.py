import pandas as pd
import math


def split_csv(file_path, output_prefix):
    # Read the CSV file
    data = pd.read_csv(file_path)

    # Calculate the size of each split
    total_rows = len(data)
    split_size = math.ceil(total_rows / 3)

    # Split the data into three parts
    for i in range(3):
        start_index = i * split_size
        end_index = min(start_index + split_size, total_rows)
        split_data = data[start_index:end_index]

        # Save each split to a new file
        output_file = f"{output_prefix}_part{i + 1}.csv"
        split_data.to_csv(output_file, index=False)
        print(f"Saved split {i + 1} to {output_file}")


if __name__ == '__main__':
    # Example usage
    file_path = "./22687585/release_train_patients/release_train_patients"
    output_prefix = "./22687585/release_train_patients/release_train_patients_split_file"
    split_csv(file_path, output_prefix)
