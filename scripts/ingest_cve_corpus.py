import sys
from pathlib import Path

# Add project root to path so we can import backend modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.app.core.embedder import Embedder
from backend.app.core.vector_store import VectorStore

# A realistic sample of CVEs with vulnerable code snippets
# In production, this would parse gigabytes of NVD JSON feeds.
SAMPLE_CVES = [
    {
        "cve_id": "CVE-2021-3281",
        "description": "Django path traversal vulnerability via archive extraction",
        "severity": 7.5,
        "language": "python",
        "vulnerable_code": """
def extract_archive(archive_path, extract_to):
    import tarfile
    with tarfile.open(archive_path) as tar:
        # VULNERABILITY: Extracting without validating paths allows path traversal
        tar.extractall(path=extract_to)
        """
    },
    {
        "cve_id": "CVE-2023-28450",
        "description": "SQL Injection in user login query",
        "severity": 9.8,
        "language": "python",
        "vulnerable_code": """
def login_user(db_cursor, username, password):
    # VULNERABILITY: String formatting directly into SQL query
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    db_cursor.execute(query)
    return db_cursor.fetchone()
        """
    },
    {
        "cve_id": "CVE-2020-11022",
        "description": "Cross-site Scripting (XSS) in jQuery HTML append",
        "severity": 6.1,
        "language": "javascript",
        "vulnerable_code": """
function renderUserProfile(userData) {
    // VULNERABILITY: Directly rendering user input as HTML
    const container = document.getElementById('profile');
    container.innerHTML = "<h1>" + userData.name + "</h1><p>" + userData.bio + "</p>";
}
        """
    },
    {
        "cve_id": "CVE-2022-23648",
        "description": "Command injection via subprocess shell=True",
        "severity": 8.8,
        "language": "python",
        "vulnerable_code": """
def ping_server(ip_address):
    import subprocess
    # VULNERABILITY: Unsanitized user input passed to shell=True
    command = "ping -c 4 " + ip_address
    result = subprocess.run(command, shell=True, capture_output=True)
    return result.stdout.decode()
        """
    },
    {
        "cve_id": "CVE-2019-11324",
        "description": "Hardcoded credentials in connection string",
        "severity": 7.5,
        "language": "python",
        "vulnerable_code": """
def connect_to_database():
    import psycopg2
    # VULNERABILITY: Hardcoded secrets in source code
    conn = psycopg2.connect(
        dbname="production_db",
        user="admin",
        password="SuperSecretPassword123!",
        host="db.internal.network"
    )
    return conn
        """
    }
]

def main():
    print("Initializing Ghost Hunter Ingestion Pipeline...")
    embedder = Embedder()
    vector_store = VectorStore()

    print(f"Loaded {len(SAMPLE_CVES)} sample CVEs for ingestion.")
    
    texts_to_embed = []
    payloads = []
    
    for cve in SAMPLE_CVES:
        # We embed the vulnerable code so we can match it against the developer's code
        texts_to_embed.append(cve["vulnerable_code"])
        
        # The payload is the metadata returned when a match is found
        payloads.append({
            "cve_id": cve["cve_id"],
            "description": cve["description"],
            "severity": cve["severity"],
            "language": cve["language"],
            "source": "nvd_sample"
        })
        
    print("Embedding CVE code snippets via CodeBERT...")
    embeddings = embedder.embed_texts(texts_to_embed)
    
    print("Inserting embeddings into Qdrant 'cve_corpus' collection...")
    vector_store.insert_cves(embeddings, payloads)
    
    print("Ingestion complete! Database is ready for Ghost Hunter retrieval.")

if __name__ == "__main__":
    main()
