

# %%
NSAMPLE = 10

# %%

# -----------------------
#       Connexion
# -----------------------

# Create Duckdb connection
import duckdb
import uuid

con = duckdb.connect(database=":memory:")
con.execute("INSTALL httpfs;")
con.execute("LOAD httpfs;")

# Get data
path_data = "https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet"
query_definition = f"SELECT * FROM read_parquet('{path_data}')"
activities = con.sql(query_definition).to_df()

activities = activities.sample(NSAMPLE)

activities = (
    activities
    .assign(id=lambda x: [str(uuid.uuid4()) for _ in range(len(x))])
    .to_dict(orient="records")
)

print(activities)

# %%
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

client_qdrant = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
    port=os.environ["QDRANT_API_PORT"],
    check_compatibility=False
)

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

# Print models
for model in models.data:
    print(f"ID: {model.id}")

# %%

from tqdm import tqdm

EMB_MODEL_NAME = "qwen3-embedding:8b"
GEN_MODEL_NAME = "gpt-oss:20b"

search_embeddings = []

for activitie in tqdm(activities, desc="Generating embeddings"):
    response = client_llmlab.embeddings.create(
        model=EMB_MODEL_NAME,
        input=activitie['label']
    )
    search_embeddings.append(
        response.data[0].embedding
    )

embedding_dim = len(search_embeddings[0])

print(
    f"✓ Embeddings generated: {len(search_embeddings)} vectors "
    f"(dimension: {embedding_dim})"
)

# %%
search_embeddings[0]


# %%

QDRANT_COLLECTION_NAME = "collection_test"
RETRIEVER_LIMIT = 5

qdrant_results_descriptions = []
qdrant_results_codes = []

for search_embedding in tqdm(search_embeddings, desc="Vector search"):
    points = client_qdrant.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        query=search_embedding,
        limit=RETRIEVER_LIMIT,
    )
    
    qdrant_results_descriptions.append(
        [point["payload"]["text"] for point in points.model_dump()["points"]]
    )
    qdrant_results_codes.append(
        [point["payload"]["code"] for point in points.model_dump()["points"]]
    )

print(
    f"✓ Vector searches completed: {len(qdrant_results_descriptions)} searches, "
    f"{len(qdrant_results_descriptions[0])} points per search"
)

# %%


SYSTEM_PROMPT = """
You are an expert in the Statistical Classification of Economic Activities in the European Community (NACE).

Your task is to classify the main economic activity of a company into the NACE 2.1 classification system, based strictly on:
- the textual description of the company's activity.
- a restricted list of candidate NACE codes and their explanatory notes

You must follow the instructions rigorously and return only the requested JSON output.
"""

USER_PROMPT_TEMPLATE = """

# Company main activity:
{activity}

# Candidate NACE codes and their explanatory notes:
{proposed_nace_descriptions}

========

# Instructions:
1. You MUST select the NACE code strictly from the provided candidate list. No external codes are allowed.
2. If multiple activities are mentioned, ONLY consider the first one.
3. If the description is unclear or insufficient to determine a classification, return:
- "nace2025": null
- "codable": false
4. The selected code MUST belong to this list:
[{proposed_nace_codes}]
5. Provide a realistic confidence score between 0.00 and 1.00 (two decimal places max).
6. Your response MUST be a valid JSON object following the schema below.

- "codable": <boolean>,
- "nace2025": <string or null>,
- "confidence": <float>

7. DO NOT include any explanation, reasoning, or additional text.

"""


# %%

user_prompts = []

for activitie, description, codes_list in zip(activities, qdrant_results_descriptions, qdrant_results_codes):
    user_prompts.append(
        USER_PROMPT_TEMPLATE.format(
            activity=activitie["label"],
            proposed_nace_descriptions="## " + "\n\n## ".join(description),
            proposed_nace_codes=codes_list
        )
    )


print(f"✓ Prompts prepared: {len(user_prompts)}")

# %% 
print(user_prompts[3])


# %% 

TEMPERATURE = 0.1

llm_responses = []

for user_prompt in tqdm(user_prompts, desc="Augmented generation!"):
    response = client_llmlab.chat.completions.create(
        model=GEN_MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=TEMPERATURE,
        response_format={"type": "json_object"}
    )

    llm_responses.append(response.choices[0].message.content)

print(
    f"✓ Augmented generation completed: {len(llm_responses)} results, "
)

# %%

llm_responses
# %%
llm_responses_parsed = []

import json


for llm_response in llm_responses:
    try:
        llm_responses_parsed.append(json.loads(llm_response))
        llm_responses_parsed = [
            {**dic, "parsed": True}
            for dic in llm_responses_parsed
        ]
    except json.JSONDecodeError:
        llm_responses_parsed.append({
            "codable": False,
            "nace2025": None,
            "confidence": 0.0,
            "parsed": False
        })

llm_responses_parsed

# %%
df_eval

# %%
import pandas as pd

rows = []
for i in range(len(llm_responses_parsed)):
    pred = llm_responses_parsed[i]
    annotation = activities[i]
    rows.append(pred | annotation)

df_eval = pd.DataFrame(rows)
df_eval 
print(f"✓ Evaluation dataset created: {len(df_eval)} rows")

# %%

df_retrieved_codes = pd.DataFrame(qdrant_results_codes)
df_retrieved_codes.columns = "retriev" + df_retrieved_codes.columns.astype(str)

df_eval = pd.concat([df_eval, df_retrieved_codes], axis=1)


# %%
retrieve_cols = [col for col in df_eval.columns if col.startswith("retriev")]
df_eval["truth_in_retriever"] = df_eval.apply(
    lambda row: row["code"] in row[retrieve_cols].values,
    axis=1
)

total_accuracy = (df_eval["code"] == df_eval["nace2025"]).mean()
total_error = 1-total_accuracy

retrieval_error = 1 - df_eval["truth_in_retriever"].mean()
generation_error = total_error - retrieval_error

print(f"Evaluation Metrics:")
print(f"- Total Accuracy: {total_accuracy:.2%}")
print(f"- Total Error Rate: {total_error:.2%}")
print(f"- Retrieval Error Rate: {retrieval_error:.2%}")
print(f"- Generation Error Rate: {generation_error:.2%}")


