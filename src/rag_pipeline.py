import requests

from src.retriever import retrieve


# ============================================================
# Configuration
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"
TOP_K = 8


# ============================================================
# Build context from retrieved chunks
# ============================================================

def build_context(results):

    context_parts = []

    for result in results:

        context_parts.append(
            f"""
Source: {result['source']}
Page: {result['page']}
Chunk: {result['chunk_id']}

Text:
{result['text']}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# Correct spelling / grammar of query
# ============================================================

def normalize_question(question):

    prompt = f"""
Rewrite the following user question into clear and grammatically correct
English for searching NHAI policy documents.

Rules:

1. Correct spelling mistakes.
2. Correct grammar.
3. Preserve the exact meaning and intent.
4. Do NOT answer the question.
5. Do NOT add new facts.
6. Return ONLY the rewritten question.
7. Keep NHAI-related terminology where possible.

User question:
{question}

Rewritten question:
"""

    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0
                }
            },
            timeout=120
        )

        response.raise_for_status()

        rewritten_question = (
            response.json()["response"]
            .strip()
        )

        if rewritten_question:
            return rewritten_question

        return question

    except requests.exceptions.RequestException:
        return question


# ============================================================
# Convert follow-up question into standalone question
# ============================================================

def contextualize_question(question, chat_history):

    if not chat_history:
        return question

    # Keep only recent conversation
    recent_history = chat_history[-6:]

    history_text = ""

    for message in recent_history:

        role = message.get(
            "role",
            "user"
        )

        content = message.get(
            "content",
            ""
        )

        history_text += (
            f"{role.capitalize()}: "
            f"{content}\n"
        )

    prompt = f"""
You are helping rewrite a user's follow-up question
for an NHAI policy retrieval system.

Use the conversation history only to understand references
such as:

- it
- they
- that rule
- this condition
- what about 170 km
- who pays for it
- what facilities are required there

Rules:

1. Rewrite the current question as a complete standalone question.
2. Preserve the user's exact meaning.
3. Use conversation history only when necessary.
4. Do NOT answer the question.
5. Do NOT add new policy facts.
6. Do NOT invent details.
7. Keep important NHAI terminology.
8. Return ONLY the rewritten question.

Conversation history:

{history_text}

Current question:

{question}

Standalone question:
"""

    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0
                }
            },
            timeout=120
        )

        response.raise_for_status()

        rewritten = (
            response.json()["response"]
            .strip()
        )

        if rewritten:
            return rewritten

        return question

    except requests.exceptions.RequestException:
        return question


# ============================================================
# Ask Ollama using retrieved policy context
# ============================================================

def ask_ollama(question, context):

    prompt = f"""
You are an NHAI Policy Assistant.

Answer the user's question using ONLY the information contained
in the provided NHAI policy context.

Rules:

1. Use only facts and rules stated in the provided context.

2. Do not invent any policy, clause, requirement, number, date,
   authority, procedure, or condition.

3. You ARE allowed to perform calculations and logical reasoning
   when the required facts or rules are explicitly available
   in the context.

4. When performing a calculation:

   - Identify the relevant policy rule.
   - Apply the rule to the user's values.
   - Show the calculation briefly.
   - Clearly state the final result.

5. A calculated result does NOT need to appear word-for-word
   in the policy if it follows directly from a rule stated
   in the context.

6. Combine information from multiple relevant chunks when necessary.

7. Ignore OCR noise, broken characters, or meaningless text.

8. Do not treat OCR noise as policy information.

9. Give a direct, clear, factual, and complete answer.

10. Do not include source filenames, page numbers, chunk numbers,
    retrieval scores, or citation lists in the answer.
    The application displays sources separately.

11. Do not use outside knowledge, general knowledge, assumptions,
    or information that is not supported by the provided context.

12. If the context truly does not contain enough information
    to answer, say exactly:

"The information is not available in the provided NHAI policy circulars."

Context:

{context}

Question:

{question}

Answer:
"""

    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1
                }
            },
            timeout=120
        )

        response.raise_for_status()

        return (
            response.json()["response"]
            .strip()
        )

    except requests.exceptions.ConnectionError:

        return (
            "ERROR: Could not connect to Ollama. "
            "Please make sure Ollama is running."
        )

    except requests.exceptions.Timeout:

        return (
            "ERROR: Ollama took too long to respond."
        )

    except requests.exceptions.RequestException as e:

        return (
            f"ERROR communicating with Ollama: {e}"
        )


# ============================================================
# Terminal application
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("NHAI POLICY RAG ASSISTANT")
    print("=" * 70)

    print("Ask as many questions as you want.")
    print("Type 'exit' or 'quit' to close the application.")

    chat_history = []

    while True:

        question = input(
            "\nAsk a question about NHAI policies: "
        ).strip()

        # ----------------------------------------------------
        # Exit
        # ----------------------------------------------------

        if question.lower() in [
            "exit",
            "quit"
        ]:

            print(
                "\nClosing NHAI Policy RAG Assistant."
            )

            break

        # ----------------------------------------------------
        # Empty query
        # ----------------------------------------------------

        if not question:

            print(
                "Please enter a valid question."
            )

            continue

        # ----------------------------------------------------
        # Contextualize query
        # ----------------------------------------------------

        standalone_question = contextualize_question(
            question,
            chat_history
        )

        search_question = standalone_question

        # ----------------------------------------------------
        # First retrieval attempt
        # ----------------------------------------------------

        results = retrieve(
            standalone_question,
            top_k=TOP_K
        )

        # ----------------------------------------------------
        # Query correction fallback
        # ----------------------------------------------------

        if not results:

            print(
                "\nNo strong match found."
            )

            print(
                "Trying query correction..."
            )

            normalized_question = normalize_question(
                standalone_question
            )

            print(
                f"Corrected query: "
                f"{normalized_question}"
            )

            if (
                normalized_question
                and normalized_question.lower()
                != standalone_question.lower()
            ):

                results = retrieve(
                    normalized_question,
                    top_k=TOP_K
                )

                search_question = (
                    normalized_question
                )

        # ----------------------------------------------------
        # No relevant chunks
        # ----------------------------------------------------

        if not results:

            answer = (
                "The information is not available in the "
                "provided NHAI policy circulars."
            )

            print("\n" + "=" * 70)
            print("RAG ANSWER")
            print("=" * 70)
            print(answer)

            chat_history.append(
                {
                    "role": "user",
                    "content": question
                }
            )

            chat_history.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

            continue

        # ----------------------------------------------------
        # Build RAG context
        # ----------------------------------------------------

        context = build_context(
            results
        )

        # ----------------------------------------------------
        # Generate answer
        # ----------------------------------------------------

        answer = ask_ollama(
            question,
            context
        )

        print("\n" + "=" * 70)
        print("RAG ANSWER")
        print("=" * 70)
        print(answer)

        # ----------------------------------------------------
        # Show search query if changed
        # ----------------------------------------------------

        if search_question != question:

            print(
                f"\nSearch query used: "
                f"{search_question}"
            )

        # ----------------------------------------------------
        # Show unique top 3 sources
        # ----------------------------------------------------

        print("\nSources used:")

        seen_sources = set()
        source_count = 0

        for result in results:

            source_key = (
                result["source"],
                result["page"]
            )

            if source_key in seen_sources:
                continue

            seen_sources.add(source_key)

            print(
                f"- {result['source']} | "
                f"Page {result['page']} | "
                f"Score {result['score']:.4f}"
            )

            source_count += 1

            if source_count >= 3:
                break

        # ----------------------------------------------------
        # Save conversation history
        # ----------------------------------------------------

        chat_history.append(
            {
                "role": "user",
                "content": question
            }
        )

        chat_history.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


# ============================================================
# Run terminal app
# ============================================================

if __name__ == "__main__":
    main()