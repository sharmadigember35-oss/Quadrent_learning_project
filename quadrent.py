from pathlib import Path

import json
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer


# ============================================================
# PART 1 — SETTINGS
# ============================================================

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not QDRANT_URL or not QDRANT_API_KEY:
    raise ValueError("Please set QDRANT_URL and QDRANT_API_KEY in your .env file.")

if not GROQ_API_KEY:
    raise ValueError("Please set GROQ_API_KEY in your .env file.")

BASE_DIR = Path(__file__).resolve().parent

# This expects a folder named "transactions" beside this Python file.
# If your folder is named "transaction" (singular), change the name below.
JSON_FOLDER = BASE_DIR / "transcripts"

# Use a new collection name so old PDF data is not mixed with JSON data.
COLLECTION_NAME = "transactions_json"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384
BATCH_SIZE = 20
TOP_K = 5


# ============================================================
# PART 2 — CONNECT TO QDRANT AND GROQ
# ============================================================

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=120,
)

groq_client = Groq(api_key=GROQ_API_KEY)


# ============================================================
# PART 3 — READ ALL JSON TRANSCRIPT FILES
# ============================================================

def load_json_transcripts(folder_path):
    """
    Read JSON files that contain video metadata and transcript segments.
    Expected shape:
    {
        "video_id": "...",
        "title": "...",
        "segments": [
            {"start": 0.0, "end": 10.0, "text": "..."}
        ]
    }
    Invalid or empty JSON files are skipped instead of stopping the program.
    """
    if not folder_path.exists():
        raise FileNotFoundError(
            f"JSON folder not found: {folder_path}\n"
            "Put the folder beside this Python file or update JSON_FOLDER."
        )

    json_files = sorted(folder_path.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json files found inside: {folder_path}")

    documents = []
    skipped_files = 0

    for json_path in json_files:
        try:
            raw_text = json_path.read_text(encoding="utf-8-sig").strip()
            if not raw_text:
                print(f"Skipping empty file: {json_path.name}")
                skipped_files += 1
                continue

            data = json.loads(raw_text)

            # Also accept a JSON file containing a list of transcript objects.
            transcript_items = data if isinstance(data, list) else [data]

            found_segment = False

            for transcript in transcript_items:
                if not isinstance(transcript, dict):
                    continue

                video_id = str(transcript.get("video_id", "")).strip()
                title = str(transcript.get("title", json_path.stem)).strip()
                segments = transcript.get("segments", [])

                if not isinstance(segments, list):
                    continue

                for segment_index, segment in enumerate(segments):
                    if not isinstance(segment, dict):
                        continue

                    text = str(segment.get("text", "")).strip()
                    if not text:
                        continue

                    start = segment.get("start", 0)
                    end = segment.get("end", start)

                    try:
                        start = float(start)
                    except (TypeError, ValueError):
                        start = 0.0

                    try:
                        end = float(end)
                    except (TypeError, ValueError):
                        end = start

                    documents.append({
                        "text": text,
                        "source": json_path.name,
                        "video_id": video_id,
                        "title": title or json_path.stem,
                        "start": start,
                        "end": end,
                        "segment_index": segment_index,
                    })
                    found_segment = True

            if not found_segment:
                print(f"Warning: no usable transcript segments in {json_path.name}")

        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as error:
            print(f"Skipping {json_path.name}: {error}")
            skipped_files += 1

    print(f"\nJSON files found: {len(json_files)}")
    print(f"Transcript segments loaded: {len(documents)}")
    print(f"Files skipped: {skipped_files}")

    if not documents:
        raise ValueError(
            "No transcript segments were loaded. Check that JSON files contain "
            "'segments' with 'text' fields."
        )

    return documents


# ============================================================
# PART 4 — CREATE / CHECK QDRANT COLLECTION
# ============================================================

def ensure_collection():
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )
        print(f"Created Qdrant collection: {COLLECTION_NAME}")
    else:
        print(f"Using existing Qdrant collection: {COLLECTION_NAME}")


# ============================================================
# PART 5 — EMBED JSON TRANSCRIPT SEGMENTS
# ============================================================

def create_embeddings(documents, model):
    texts = [document["text"] for document in documents]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    )

    return embeddings


# ============================================================
# PART 6 — UPLOAD TO QDRANT IN SMALL BATCHES
# ============================================================

def upload_documents(documents, embeddings):
    total = len(documents)

    for start_index in range(0, total, BATCH_SIZE):
        end_index = min(start_index + BATCH_SIZE, total)
        points = []

        for index in range(start_index, end_index):
            document = documents[index]

            # Stable IDs mean rerunning this script updates the same transcript
            # segment rather than creating a new point for that segment.
            stable_key = (
                f"{document['source']}|{document['video_id']}|"
                f"{document['segment_index']}|{document['start']}"
            )
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, stable_key))

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embeddings[index].tolist(),
                    payload={
                        "text": document["text"],
                        "source": document["source"],
                        "video_id": document["video_id"],
                        "title": document["title"],
                        "start": document["start"],
                        "end": document["end"],
                        "segment_index": document["segment_index"],
                    },
                )
            )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

        print(f"Uploaded {end_index}/{total} transcript segments")


# ============================================================
# PART 7 — SEARCH SIMILAR JSON CONTENT
# ============================================================

def search(query, model, top_k=1):
    query_vector = model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points

    return results


# ============================================================
# PART 8 — ASK GROQ USING RETRIEVED TRANSCRIPT SEGMENTS
# ============================================================

def format_timestamp(seconds):
    seconds = max(0, int(float(seconds)))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def ask_llm(question, results):
    if not results:
        return "I don't know based on the provided JSON transcripts."

    context_parts = []

    for result in results:
        payload = result.payload or {}
        text = payload.get("text", "")
        title = payload.get("title", "Unknown title")
        video_id = payload.get("video_id", "")
        start = format_timestamp(payload.get("start", 0))
        end = format_timestamp(payload.get("end", 0))
        source = payload.get("source", "Unknown JSON file")

        if text:
            video_url = (
                f"https://www.youtube.com/watch?v={video_id}&t="
                f"{int(float(payload.get('start', 0)))}s"
                if video_id else "No video ID available"
            )
            context_parts.append(
                f"JSON file: {source}\n"
                f"Video title: {title}\n"
                f"Video ID: {video_id}\n"
                f"Timestamp: {start}–{end}\n"
                f"Video URL: {video_url}\n"
                f"Transcript text: {text}"
            )

    context = "\n\n".join(context_parts)

    prompt = f"""
You answer questions using only the retrieved JSON transcript segments.

Rules:
- Use only the context below. Do not invent information.
- Answer the user's question clearly and simply.
- If relevant, include the video title, timestamp, and YouTube URL.
- If the answer is not supported by the context, say:
  "I don't know based on the provided JSON transcripts."

Retrieved context:
{context}

Question:
{question}
"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    return response.choices[0].message.content or ""


# ============================================================
# PART 9 — MAIN PROGRAM
# ============================================================

def main():
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Embedding model ready.")

    ensure_collection()

    documents = load_json_transcripts(JSON_FOLDER)
    embeddings = create_embeddings(documents, model)

    print(f"Generated {len(embeddings)} embeddings")
    upload_documents(documents, embeddings)

    print("\nJSON RAG is ready. Type 'exit' to stop.")

    while True:
        question = input("\nAsk a question about your transcripts: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            results = search(question, model)

            print("\nRetrieved transcript segments:")
            for result in results:
                payload = result.payload or {}
                print(
                    f"\nScore: {result.score:.3f} | "
                    f"Title: {payload.get('title', 'Unknown')} | "
                    f"Time: {format_timestamp(payload.get('start', 0))}"
                )
                print(payload.get("text", ""))

            print("\nFinal Answer:")
            print(ask_llm(question, results))

        except Exception as error:
            print(f"Could not answer this question: {error}")



# ============================================================
# PART 9 — TEST OUR RAG SYSTEM
# ============================================================

def check_refusal(answer):
    """Return True if the model says it cannot answer from our data."""

    answer = answer.lower()

    refusal_phrases = [
        "i don't know based on the provided json transcripts",
        "i do not know based on the provided json transcripts",
        "not supported by the context",
        "not present in the context"
    ]

    for phrase in refusal_phrases:
        if phrase in answer:
            return True

    return False


def run_evaluation(model):
    # Read our test questions.
    test_file = BASE_DIR / "evaluation_cases.json"

    with test_file.open("r", encoding="utf-8") as file:
        test_cases = json.load(file)

    total = len(test_cases)
    passed = 0
    failed = 0
    reports = []

    print("\n========== RAG EVALUATION ==========")

    for number, case in enumerate(test_cases, start=1):
        question = case["q"]

        print(f"\nTest {number}/{total}: {question}")

        try:
            # 1. Search Qdrant for the top 5 segments.
            results = search(question, model, top_k=5)

            # 2. Ask Groq using the retrieved segments.
            answer = ask_llm(question, results)

            # 3. Check the result.
            if case.get("expect_refusal", False):

                # This question should be outside our transcript scope.
                test_passed = check_refusal(answer)
                reason = (
                    "Correctly refused"
                    if test_passed
                    else "Expected a refusal, but got an answer"
                )

            else:
                expected_video = case["expect_video_id"]
                expected_time = case["expect_around_sec"]
                tolerance = case.get("tolerance_sec", 120)

                test_passed = False

                # Look for the expected video near the expected timestamp.
                for result in results:
                    payload = result.payload or {}

                    video_id = payload.get("video_id")
                    start = payload.get("start")

                    if video_id == expected_video and start is not None:
                        time_difference = abs(
                            float(start) - expected_time
                        )

                        if time_difference <= tolerance:
                            test_passed = True
                            break

                reason = (
                    "Expected video and timestamp found"
                    if test_passed
                    else "Expected video/timestamp not found in top 5"
                )

            # 4. Display PASS or FAIL.
            if test_passed:
                passed += 1
                status = "PASS"
            else:
                failed += 1
                status = "FAIL"

            print(f"Result: {status}")
            print(f"Reason: {reason}")

            # 5. Show retrieved videos to help debug failures.
            print("Retrieved results:")

            retrieved_details = []

            for result in results:
                payload = result.payload or {}

                details = {
                    "video_id": payload.get("video_id"),
                    "title": payload.get("title"),
                    "start": payload.get("start"),
                    "score": result.score,
                    "text": payload.get("text", "")
                }

                retrieved_details.append(details)

                print(
                    f"  Video: {details['video_id']} | "
                    f"Time: {format_timestamp(details['start'] or 0)} | "
                    f"Score: {details['score']:.3f}"
                )

            reports.append({
                "question": question,
                "status": status,
                "reason": reason,
                "answer": answer,
                "retrieved": retrieved_details
            })

        except Exception as error:
            failed += 1

            print("Result: ERROR")
            print(f"Error: {error}")

            reports.append({
                "question": question,
                "status": "ERROR",
                "error": str(error)
            })

    # 6. Save the complete report.
    report_file = BASE_DIR / "evaluation_results.json"

    with report_file.open("w", encoding="utf-8") as file:
        json.dump(reports, file, indent=2, ensure_ascii=False)

    print("\n========== FINAL SUMMARY ==========")
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if total > 0:
        print(f"Pass rate: {passed / total * 100:.2f}%")

    print(f"Detailed report saved to: {report_file}")

# ============================================================
# PART 10 — MAIN PROGRAM
# ============================================================

def main():
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Embedding model ready.")

    ensure_collection()

    print("\nChoose a mode:")
    print("1. Run test cases")
    print("2. Ask questions manually")

    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        # Uses the existing Qdrant collection.
        # Does not upload or delete any points.
        run_evaluation(model)
        return

    # Existing indexing and interactive flow.
    documents = load_json_transcripts(JSON_FOLDER)
    embeddings = create_embeddings(documents, model)

    print(f"Generated {len(embeddings)} embeddings")
    upload_documents(documents, embeddings)

    print("\nJSON RAG is ready. Type 'exit' to stop.")

    while True:
        question = input(
            "\nAsk a question about your transcripts: "
        ).strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            results = search(question, model)

            print("\nRetrieved transcript segments:")

            for result in results:
                payload = result.payload or {}

                print(
                    f"\nScore: {result.score:.3f} | "
                    f"Title: {payload.get('title', 'Unknown')} | "
                    f"Time: {format_timestamp(payload.get('start', 0))}"
                )

                print(payload.get("text", ""))

            print("\nFinal Answer:")
            print(ask_llm(question, results))

        except Exception as error:
            print(f"Could not answer this question: {error}")


 



if __name__ == "__main__":
    main()

 
