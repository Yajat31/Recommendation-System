import requests
import json
import time
import logging
import argparse
from elasticsearch import Elasticsearch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SIMULATOR_URL = "http://localhost:3000"
ES_HOST = "http://localhost:9200"
INDEX_NAME = "articles"
LOG_FILE = "interaction_logs.jsonl"

def get_query():
    try:
        response = requests.get(f"{SIMULATOR_URL}/query")
        if response.status_code == 200:
            return response.json()
        else:
            logging.error(f"Failed to get query: {response.status_code}")
            return None
    except Exception as e:
        logging.error(f"Error getting query: {e}")
        return None

def search_es(es_client, query_text, size=10):
    body = {
        "query": {
            "match": {
                "text": query_text
            }
        },
        "size": size
    }
    try:
        response = es_client.search(index=INDEX_NAME, body=body)
        hits = response['hits']['hits']
        return [hit['_id'] for hit in hits]
    except Exception as e:
        logging.error(f"Error searching ES: {e}")
        return []

def submit_ranklist(query_id, user_id, ranked_article_ids):
    payload = {
        "query_id": query_id,
        "user_id": user_id,
        "ranked_article_ids": ranked_article_ids
    }
    try:
        response = requests.post(f"{SIMULATOR_URL}/ranklist", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            logging.error(f"Failed to submit ranklist: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logging.error(f"Error submitting ranklist: {e}")
        return None

def log_interaction(query_data, ranked_ids, actions):
    log_entry = {
        "query_id": query_data['query_id'],
        "user_id": query_data['user_id'],
        "query_text": query_data['query_text'],
        "ranked_article_ids": ranked_ids,
        "actions": actions
    }
    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps(log_entry) + "\n")

def main():
    parser = argparse.ArgumentParser(description='Collect user interactions from simulator')
    parser.add_argument('-n', '--num-iterations', type=int, default=100,
                       help='Number of iterations to run (default: 100)')
    args = parser.parse_args()
    
    es = Elasticsearch(hosts=[ES_HOST])
    if not es.ping():
        logging.error("Elasticsearch is not reachable.")
        return

    logging.info(f"Starting simulation loop ({args.num_iterations} iterations)...")
    
    # Run for a fixed number of iterations
    for i in range(args.num_iterations):
        query_data = get_query()
        if not query_data:
            time.sleep(1)
            continue
            
        logging.info(f"Iteration {i+1}: Processing query '{query_data['query_text']}'")
        
        # Search ES
        # Note: We might need to search multiple fields. For now, just headline.
        # Let's inspect the article structure first to be sure.
        ranked_ids = search_es(es, query_data['query_text'], size=10)
        
        if not ranked_ids:
            logging.warning("No results found for query.")
            # Even if no results, we might need to send something? 
            # Or maybe just skip. The simulator expects a list.
            # If empty, we probably get empty actions.
        
        # Submit ranklist
        result = submit_ranklist(query_data['query_id'], query_data['user_id'], ranked_ids)
        
        if result:
            actions = result.get('actions', [])
            log_interaction(query_data, ranked_ids, actions)
            logging.info(f"Logged interaction for user {query_data['user_id']}")
            
        time.sleep(0.1) # Be nice to the simulator

if __name__ == "__main__":
    main()
