import json
import fire

def average_stats(json_path, urls_path):
    """
    Computes the average of statistics over all JSON entries whose 'url' is present in the URLs file.
    
    Parameters:
      json_path (str): Path to the JSON file.
      urls_path (str): Path to a text file containing one URL per line.
    """
    # Load JSON data from the file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Load URLs from the text file (ignoring empty lines)
    with open(urls_path, 'r') as f:
        url_list = [line.strip() for line in f if line.strip()]
    url_list = [url[:-1] if url.endswith("/") else url for url in url_list]
    url_set = set(url_list)

    # Initialize accumulators and counter
    total_fully_supervised_ap = 0.0
    total_teacher_ap = 0.0
    total_student_ap = 0.0
    total_label_percentage = 0.0
    count = 0
    
    # Iterate over each record in the JSON
    for record in data.values():
        url = record.get("url")
        if url in url_set:
            total_fully_supervised_ap += record.get("fully_supervised_ap", 0.0)
            total_teacher_ap += record.get("teacher_ap", 0.0)
            total_student_ap += record.get("student_ap", 0.0)
            total_label_percentage += record.get("label_percentage", 0.0)
            count += 1
    
    if count == 0:
        print("No matching URLs found in the JSON data.")
        return
    
    # Calculate overall averages
    averages = {
        "fully_supervised_ap": total_fully_supervised_ap / count,
        "teacher_ap": total_teacher_ap / count,
        "student_ap": total_student_ap / count,
        "label_percentage": total_label_percentage / count,
        "count": count  # Optional: number of entries used for the average
    }
    
    # Output the averages as formatted JSON
    print(json.dumps(averages, indent=2))

if __name__ == '__main__':
    fire.Fire(average_stats)