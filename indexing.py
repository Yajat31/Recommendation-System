import json
import logging
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_articles(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            yield json.loads(line)

def index_articles(es_client, index_name, file_path):
    logging.info(f"Creating index '{index_name}'...")
    if es_client.indices.exists(index=index_name):
        logging.info(f"Index '{index_name}' already exists. Skipping creation.")
    else:
        es_client.indices.create(index=index_name)
        logging.info(f"Index '{index_name}' created.")

    logging.info("Indexing articles...")
    
    actions = []
    batch_size = 1000
    count = 0

    for article in tqdm(load_articles(file_path), desc="Reading articles"):
        action = {
            "_index": index_name,
            "_id": article['uuid'],
            "_source": article
        }
        actions.append(action)
        
        if len(actions) >= batch_size:
            helpers.bulk(es_client, actions)
            count += len(actions)
            actions = []
    
    if actions:
        helpers.bulk(es_client, actions)
        count += len(actions)
        
    logging.info(f"Indexed {count} articles.")

if __name__ == "__main__":
    ES_HOST = "http://localhost:9200"
    INDEX_NAME = "articles"
    ARTICLES_FILE = "artifacts/articles.jsonl"

    es = Elasticsearch(hosts=[ES_HOST])
    
    # Wait for ES to be ready
    if not es.ping():
        logging.error("Elasticsearch is not reachable.")
        exit(1)
        
    index_articles(es, INDEX_NAME, ARTICLES_FILE)
