import os
from dotenv import load_dotenv
import anthropic
import time
import rag

load_dotenv(override=True)

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL"),
)

def extract_text(response):
    parts = []
    for block in response.content:
        if block.type == "text":
            parts.append(block.text)
    return "".join(parts)

MAX_RETRIES = 4

def stream_reply(messages):
    """Stream Claude's reply, retrying on transient errors. Same pattern as Project 1."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print("\nAssistant: ", end="", flush=True)
            with client.messages.stream(
                model="claude-sonnet-5",
                max_tokens=1024,
                messages=messages,
            ) as stream:
                for text_chunk in stream.text_stream:
                    print(text_chunk, end="", flush=True)
                final_message = stream.get_final_message()
            print("\n")
            return final_message

        except (
            anthropic.APIConnectionError,
            anthropic.RateLimitError,
            anthropic.InternalServerError,
        ) as error:
            wait_seconds = 2 ** (attempt - 1)
            print(
                f"\n[Transient error: {type(error).__name__}. "
                f"Retry {attempt}/{MAX_RETRIES} in {wait_seconds}s...]"
            )
            time.sleep(wait_seconds)

    print("[Failed after several retries. Please try your question again.]")
    return None


chunks = rag.load_index("index.json")
messages = []
total_input_tokens = 0
total_output_tokens = 0

print("SAP Policy Assistant. Type your question, or 'exit' to quit.\n")

while True:
    question = input("You: ")

    if question.lower() in ("exit", "quit"):
        print("Goodbye!")
        break

    top_results = rag.search(question, chunks, top_k=3)
    prompt = rag.build_prompt(question, top_results)
    api_messages = messages + [{"role": "user", "content": prompt}]

    final_message = stream_reply(api_messages)

    if final_message is None:
        continue  # all retries failed — don't touch history, let the user try again

    answer = extract_text(final_message)
    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": answer})

    input_tokens = final_message.usage.input_tokens
    output_tokens = final_message.usage.output_tokens
    total_input_tokens += input_tokens
    total_output_tokens += output_tokens
    print(
        f"[tokens] this turn: {input_tokens} in / {output_tokens} out | "
        f"session total: {total_input_tokens} in / {total_output_tokens} out"
    )