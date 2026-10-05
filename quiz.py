"""
Quiz generation for the personalized learning module.
"""

import json
import re


QUIZ_PROMPT = """
You are a quiz generator for a personalized learning platform.

Generate multiple-choice questions ONLY from the provided study material.

For every question provide:
- question
- four options: A, B, C, D
- correct answer
- topic

The topic should be a short concept name such as:
"Deadlock", "CPU Scheduling", "Memory Management", etc.

Return ONLY valid JSON.

Expected format:

[
    {
        "question": "Question text",
        "options": {
            "A": "Option A",
            "B": "Option B",
            "C": "Option C",
            "D": "Option D"
        },
        "answer": "A",
        "topic": "Topic name"
    }
]

Study material:
{context}

Generate {num_questions} questions.
"""


def generate_quiz(context, llm, num_questions=10):
    """
    Generate quiz questions in batches.

    Large quizzes are divided into smaller batches so that
    the LLM does not produce incomplete JSON.
    """

    all_questions = []

    batch_size = 10
    remaining = num_questions

    while remaining > 0:

        current_batch = min(batch_size, remaining)

        prompt = QUIZ_PROMPT.replace("{context}", context)
        prompt = prompt.replace(
            "{num_questions}",
            str(current_batch)
        )

        response = llm.invoke([prompt])

        content = response.content.strip()

        # Remove markdown code fences if the LLM adds them.
        content = re.sub(r"```json\s*", "", content)
        content = re.sub(r"```\s*", "", content)
        content = content.strip()

        try:
            quiz_batch = json.loads(content)

        except json.JSONDecodeError as e:
            raise ValueError(
                f"Could not parse quiz response as JSON: {e}\n"
                f"LLM response:\n{content}"
            )

        if not isinstance(quiz_batch, list):
            raise ValueError(
                "Quiz response must be a JSON list."
            )

        all_questions.extend(quiz_batch)

        remaining -= current_batch

    return all_questions[:num_questions]