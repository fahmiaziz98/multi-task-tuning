import argparse
from inference import QuizGenerator


TEST_CONTEXTS = [
    "The mitochondria is the organelle responsible for producing ATP, "
    "the main energy currency of the cell.",

    "The Great Wall of Manado, a lesser-known coastal fortification built "
    "in 1673 by Dutch colonial forces, stretches approximately 4 kilometers "
    "along the northern shoreline of Sulawesi.",

    "Studio Ghibli was co-founded in 1985 by directors Hayao Miyazaki and "
    "Isao Takahata, along with producer Toshio Suzuki. The studio is based "
    "in Koganei, Tokyo, and is known for hand-drawn animated films such as "
    "Spirited Away and My Neighbor Totoro.",

    "In distributed systems, the CAP theorem states that a distributed "
    "data store can only provide two of the following three guarantees "
    "simultaneously: Consistency, Availability, and Partition tolerance. "
    "This tradeoff was first articulated by computer scientist Eric Brewer "
    "in 2000 and later formally proven by Seth Gilbert and Nancy Lynch in 2002.",

    "The Voyager 1 spacecraft was launched by NASA on September 5, 1977, "
    "with a primary mission to study the outer Solar System, including "
    "close flybys of Jupiter and Saturn. After completing its planetary "
    "objectives, Voyager 1 continued into interstellar space, crossing the "
    "heliopause in August 2012 and becoming the first human-made object to "
    "leave the heliosphere. As of its most recent status reports, the "
    "probe continues to transmit data back to Earth using a radioisotope "
    "thermoelectric generator, despite being over 24 billion kilometers "
    "from the Sun.",
]


def run_tests(qa_pair_checkpoint: str, distractor_checkpoint: str) -> None:
    """Generate a quiz for each test context and print the result.

    Args:
        qa_pair_checkpoint: Local path or HF Hub repo id of the qa_pair model.
        distractor_checkpoint: Local path or HF Hub repo id of the
            distractor model.
    """
    generator = QuizGenerator(
        qa_pair_checkpoint=qa_pair_checkpoint,
        distractor_checkpoint=distractor_checkpoint,
    )

    for i, context in enumerate(TEST_CONTEXTS, start=1):
        quiz = generator.generate_quiz(context=context)
        print(f"\n{'=' * 70}")
        print(f"Test {i} (context length: {len(context)} chars)")
        print(f"{'=' * 70}")
        print(f"Context:     {context}")
        print(f"Question:    {quiz.question}")
        print(f"Answer:      {quiz.answer}")
        print(f"Distractors: {quiz.distractors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test QuizGenerator on varied contexts.")
    parser.add_argument("--qa_pair_checkpoint", type=str, required=True)
    parser.add_argument("--distractor_checkpoint", type=str, required=True)
    args = parser.parse_args()

    run_tests(args.qa_pair_checkpoint, args.distractor_checkpoint)
