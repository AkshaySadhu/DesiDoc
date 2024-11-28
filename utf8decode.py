import json

def replace_non_utf8_bytes(file_content):
    """
    Replaces non-UTF-8 bytes in the file content by decoding with 'replace' to preserve readable content.
    """
    return file_content.decode("utf-8", "replace")

def process_json_file_limit(input_file, output_file, limit=500000):
    """
    Processes the input JSON file with non-UTF-8 bytes, selects up to `limit` entries,
    and saves the cleaned data to a new JSON file.
    """
    try:
        # Read the file in binary mode to handle invalid UTF-8 bytes
        with open(input_file, "rb") as file:
            file_content = file.read()

        # Decode the content while replacing invalid bytes
        decoded_content = replace_non_utf8_bytes(file_content)

        # Parse the decoded content as JSON
        data = json.loads(decoded_content)

        # Limit the number of entries to process
        limited_data = data[:limit]

        # Save the limited data to the output JSON file
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(limited_data, file, ensure_ascii=False, indent=4)

        print(f"Processed {len(limited_data)} entries and saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    # File paths
    input_file = "./22687585/release_train_patients/training_data_4.json"  # Replace with your actual input file name
    output_file = "./22687585/release_train_patients/training_data_4_limited.json"  # Output file name

    # Process the JSON file, limiting to 500,000 entries
    process_json_file_limit(input_file, output_file, limit=500000)
