# %%
import uuid

# %%

# -----------------------
#       Connexions
# -----------------------

# Create Duckdb connection
import duckdb
con = duckdb.connect(database=":memory:")
con.execute("INSTALL httpfs;")
con.execute("LOAD httpfs;")


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

# llm.lab client
from openai import OpenAI

client_llmlab = OpenAI(
    base_url=os.environ["LLMLAB_URL"],
    api_key=os.environ["LLMLAB_API_KEY"],
)

models = client_llmlab.models.list()

# Afficher la liste des modèles
for model in models.data:
    print(f"ID: {model.id}")

# %%

# Embedding model parameters
emb_dim = 4096
emb_model = "qwen3-embedding:8b"

# %%


# ----------------------
#      Get NACE 2.1
# ----------------------

path_nace = "../NACE_Rev2.1_Structure_Explanatory_Notes_EN.tsv"
query_definition = f"SELECT * FROM read_csv('{path_nace}')"
nace_df = con.sql(query_definition).to_df()
nace_df.head()

# %%
nace_df_correct = nace_df.fillna("")
nace = nace_df_correct.to_dict(orient="records")
nace

# %%

## NaceDocument 

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class naceDocument:
    raw: Dict[str, Any]

    # Normalized fields
    code: str = field(init=False)
    heading: str = field(init=False)
    level: int = field(init=False)
    parent_code: Optional[str] = field(init=False)

    # description: str = field(init=False)
    includes: Optional[str] = field(init=False)
    includes_also: Optional[str] = field(init=False)
    excludes: Optional[str] = field(init=False)

    def __post_init__(self):
        self._validate_and_normalize()

    # --------------------------------------------------
    # VALIDATION & NORMALIZATION
    # --------------------------------------------------
    def _validate_and_normalize(self):
        required_fields = ["CODE", "HEADING", "LEVEL"]

        for field_name in required_fields:
            if field_name not in self.raw or self.raw[field_name] in [None, ""]:
                raise ValueError(f"Missing required field: {field_name}")

        self.code = str(self.raw["CODE"]).strip()
        self.heading = self._clean_text(self.raw["HEADING"])
        self.level = int(self.raw["LEVEL"])

        self.parent_code = self._safe_get("PARENT_CODE")

        #self.description = self._clean_text(self._safe_get("Includes"))
        self.includes = self._clean_text(self._safe_get("Includes"))
        self.includes_also = self._clean_text(self._safe_get("IncludesAlso"))
        self.excludes = self._clean_text(self._safe_get("Excludes"))

        self._sanity_checks()

    def _safe_get(self, key: str) -> Optional[str]:
        value = self.raw.get(key)
        if value is None:
            return None
        value = str(value).strip()
        return value if value != "" else None

    def _clean_text(self, text: Optional[str]) -> Optional[str]:
        if text is None:
            return None
        return " ".join(text.replace("\n", " ").split())

    def _sanity_checks(self):
        # Basic structural validation
        if self.level < 1 or self.level > 4:
            raise ValueError(f"Invalid level {self.level} for code {self.code}")

        # Code consistency check
        if (self.parent_code) and (self.level > 2):
            if not self.code.startswith(self.parent_code[: len(self.parent_code)]):
                # Not strict but useful warning
                print(f"[Warning] Code {self.code} not aligned with parent {self.parent_code}")

    # --------------------------------------------------
    # TEXT GENERATION FOR EMBEDDING
    # --------------------------------------------------
    def to_embedding_text(
        self,
        include_includes: bool = True,
        include_includes_also: bool = False,
        include_excludes: bool = False,
        include_hierarchy: bool = False,
        parent_chain: Optional[List[str]] = None,
    ) -> str:
        """
        Generate embedding-ready text with configurable components.
        """

        parts = []

        # Core identity
        parts.append(f"Code: {self.code}")
        parts.append(f"Title: {self.heading}")

        # Optional hierarchy context
        if include_hierarchy and parent_chain:
            parts.append("Hierarchy: " + " > ".join(parent_chain))

        # Main description
        # if self.description:
        #     parts.append(f"Description: {self.description}")

        # Includes
        if include_includes and self.includes:
            parts.append(f"Includes: {self.includes}")

        if include_includes_also and self.includes_also:
            parts.append(f"Also includes: {self.includes_also}")

        # Excludes
        if include_excludes and self.excludes:
            parts.append(f"Excludes: {self.excludes}")

        return "\n".join(parts)

    # --------------------------------------------------
    # EXPORT FOR VECTOR DB (QDRANT)
    # --------------------------------------------------
    def to_qdrant_payload(
        self,
        embedding_text: str,
    ) -> Dict[str, Any]:
        """
        Returns payload ready for Qdrant ingestion.
        Vector must be added separately.
        """

        return {
            "id": self.code.replace(".", "00000"),
            "payload": {
                "code": self.code,
                "heading": self.heading,
                "level": self.level,
                "parent_code": self.parent_code,
                "text": embedding_text,
            }
        }




# %%

# --------------------------------------------
#      Get structured text for each code
# --------------------------------------------

payloads = []

for code_dict in nace:

    doc = naceDocument(code_dict)

    text = doc.to_embedding_text(
        include_includes=True,
        include_excludes=True,
        include_hierarchy=True,
    )

    payloads.append(
        doc.to_qdrant_payload(text)
    ) 


# %%
print(payloads[12]["payload"]["text"])


# %%

from qdrant_client.models import Distance, VectorParams, PointStruct

qdrant_collection_name = "collection_test"
client_qdrant.recreate_collection(
    collection_name=qdrant_collection_name,
    vectors_config=VectorParams(
        size=emb_dim,
        distance=Distance.COSINE
    )
)
# %%

# Get embeddings :

embeddings = []

for i, payload in enumerate(payloads):
    try:
        response = client_llmlab.embeddings.create(
            model=emb_model,
            input=payload["payload"]["text"]
        )
        embeddings.append(response.data[0].embedding)
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(payloads)} embeddings")
    
    except Exception as e:
        print(f"Failed to generate embedding for document {i}: {str(e)}")
        continue
# %%

# Create structured points : 

points = []
for i, (payload, embedding) in enumerate(zip(payloads, embeddings)):
    points.append(
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "text": payload["payload"]["text"],
                **payload["payload"]
            }
        )
    )
# %%
points[100]

# %%
upload_batch_size = 10

for i in range(0, len(points), upload_batch_size):
    batch = points[i:i + upload_batch_size]
    try:
        client_qdrant.upsert(
            collection_name=qdrant_collection_name,
            points=batch
        )
        print(f"Uploaded batch {i//upload_batch_size + 1}/{(len(points)-1)//upload_batch_size + 1}")
    except Exception as e:
        print(f"Failed to upload batch {i//upload_batch_size + 1}: {str(e)}")
        continue

# %%
