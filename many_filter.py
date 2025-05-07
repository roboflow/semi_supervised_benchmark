import json
import fire


def slow_get_category(dataset_mapping, key):
    short_key = "-".join(key.split("-")[:-6])
    for ds_key in dataset_mapping.keys():
        if short_key in ds_key:
            return dataset_mapping[ds_key]
    return None


def average_stats(json_path, urls_path, mapping_path, model_filter=""):
    """
    Computes the average statistics per dataset group (category) and overall across all JSON entries
    that meet the following conditions:
    
    1. The 'url' value is present in the provided URLs file.
    2. The JSON record's key contains the given model_filter string (if provided).
    3. The JSON record's key starts with one of the dataset names specified in the mapping file.
    
    Parameters:
      json_path (str): Path to the JSON file with dataset results.
      urls_path (str): Path to a text file containing one URL per line.
      mapping_path (str): Path to the JSON file mapping dataset names to categories.
      model_filter (str): A string used to filter dataset keys by model variant.
                          Only records with keys containing this string will be considered.
                          Defaults to an empty string (no filtering).
    """
    # Load results JSON data
    with open(json_path, 'r') as f:
        results_data = json.load(f)
    
    # Load URLs from file (ignoring empty lines) and normalize URLs without trailing slash
    with open(urls_path, 'r') as f:
        url_list = [line.strip() for line in f if line.strip()]
    url_list = [url[:-1] if url.endswith("/") else url for url in url_list]
    url_set = set(url_list)
    
    # Load dataset-to-category mapping JSON
    with open(mapping_path, 'r') as f:
        dataset_mapping = json.load(f)
    
    # Prepare dictionaries to accumulate statistics per category and overall
    category_stats = {}
    overall_stats = {
        "fully_supervised_ap": 0.0,
        "fully_supervised_ap_50": 0.0,
        "teacher_ap": 0.0,
        "student_ap": 0.0,
        "label_percentage": 0.0,
        "count": 0
    }
    
    # Iterate over each record in the results JSON
    for key, record in results_data.items():
        # If model_filter is provided, skip keys that do not contain the filter string.
        if model_filter and model_filter not in key or "fsod" not in key or "federated" not in key:
            continue
        
        # Check if the record's URL is in the provided URL list
        record_url = record.get("url", "")
        if record_url.endswith("/"):
            record_url = record_url[:-1]
        if record_url not in url_set:
            print(record_url)
            continue
            
        
        # Determine the dataset name by finding a mapping key that the record key starts with.
        dataset_name = None
        for ds_key in dataset_mapping.keys():
            short_key = "-".join(key.split("-")[:-6])
            if ds_key.startswith(short_key):
                dataset_name = ds_key
                break
        
        # If no matching dataset name is found, skip this record.
        if dataset_name is None:
            continue
        
        # Get the category for this dataset name.
        category = dataset_mapping.get(dataset_name)
        # category = slow_get_category(dataset_mapping, key)
        # print(key, category)
        
        # Initialize accumulator for the category if it doesn't exist.
        if category not in category_stats:
            category_stats[category] = {
                "fully_supervised_ap": 0.0,
                "fully_supervised_ap_50": 0.0,
                "teacher_ap": 0.0,
                "student_ap": 0.0,
                "label_percentage": 0.0,
                "count": 0
            }
        
        # Accumulate the values for category and overall
        for stat in ["fully_supervised_ap", "fully_supervised_ap_50", "teacher_ap", "student_ap", "label_percentage"]:
            value = record.get(stat, 0.0)
            category_stats[category][stat] += value
            overall_stats[stat] += value
        
        category_stats[category]["count"] += 1
        overall_stats["count"] += 1
    
    # Compute average statistics for each category.
    averages = {}
    for category, values in category_stats.items():
        count = values.pop("count")
        if count > 0:
            averages[category] = {stat: total / count for stat, total in values.items()}
            averages[category]["count"] = count  # Optional: include count of records used
        else:
            averages[category] = {}
    
    # Compute overall averages if any records were processed.
    overall_count = overall_stats.pop("count")
    if overall_count > 0:
        overall_averages = {stat: total / overall_count for stat, total in overall_stats.items()}
        overall_averages["count"] = overall_count  # Optional: include count of records used
    else:
        overall_averages = {}
    
    # Add the overall statistics under the "overall" key
    averages["overall"] = overall_averages
    
    # Output the averages as formatted JSON
    print(json.dumps(averages, indent=2))

if __name__ == '__main__':
    fire.Fire(average_stats)