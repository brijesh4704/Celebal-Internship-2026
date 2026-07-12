import cohere


class Chatbot:

    def __init__(self, vectorstore, cohere_api_key):

        self.vectorstore = vectorstore

        # Cohere SDK v7
        self.co = cohere.ClientV2(
            api_key=cohere_api_key
        )

    def respond(self, user_message):

        try:

            # Retrieve relevant chunks from FAISS
            retrieved_docs = self.vectorstore.retrieve(
                user_message,
                top_k=3
            )

            # Build context
            context = "\n\n".join(
                doc["text"] for doc in retrieved_docs
            )

            response = self.co.chat(
                model="command-a-03-2025",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Answer ONLY using the provided document context. "
                            "If the answer is not present in the document, say "
                            "'I couldn't find that information in the uploaded document.'"
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"""
Context:
{context}

Question:
{user_message}
""",
                    },
                ],
            )

            answer = response.message.content[0].text

            return answer, retrieved_docs

        except Exception as e:
            return f"Error: {e}", []