# %%

# Create Duckdb connection
import duckdb
con = duckdb.connect(database=":memory:")
con.execute("INSTALL httpfs;")
con.execute("LOAD httpfs;")

# Get data
path_data = "https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet"
query_definition = f"SELECT * FROM read_parquet('{path_data}')"
annotations = con.sql(query_definition).to_df()
annotations.head()


# %%
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

client_qdrant = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
    port=os.environ["QDRANT_API_PORT"]
)
# %%

collections = client_qdrant.get_collections()


for collection in collections.collections:
    print(collection.name)
# %%
