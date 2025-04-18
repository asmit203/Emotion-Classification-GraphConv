
import json
import nltk
from vllm import LLM, SamplingParams # type: ignore

# Ensure necessary nltk resources are downloaded
nltk.download("punkt")

# Initialize Qwen2.5-14B model using vLLM
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"`
llm = LLM(MODEL_NAME)
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=10)

def classify_clause_with_llm(utterance, clause, emotion):
    """
    Uses Qwen2.5-14B to classify a given clause as an emotion clause or cause clause.
    """
    prompt = (
        f"Utterance: '{utterance}'\n"
        f"Clause: '{clause}'\n"
        f"Emotion: {emotion}\n"
        "Is this an emotion clause, a cause clause, or a neutral clause?"
    )
    
    # Generate response using the model
    output = llm.generate([prompt], sampling_params)[0].outputs[0].text.strip().lower()
    
    if "emotion clause" in output:
        return "emotion_clause"
    elif "cause clause" in output:
        return "cause_clause"
    else:
        return "neutral"

def process_dataset(input_data):
    """
    Processes the dataset, classifying clauses into emotion clauses and cause clauses.
    """
    results = {}

    for conv_id, conversation in input_data.items():
        results[conv_id] = []
        
        for turn_group in conversation:  # Iterate over list of turns
            for turn in turn_group:  # Iterate over individual turns
                if not isinstance(turn, dict):
                    continue  # Skip invalid entries
                
                utterance = turn.get("utterance", "")
                emotion = turn.get("emotion", "unknown")
                cause_spans = turn.get("expanded emotion cause span", [])
                clauses = turn.get("clauses", [])

                classified_clauses = []
                emotion_clauses = []
                cause_clauses = []

                for clause in clauses:
                    label = classify_clause_with_llm(utterance, clause, emotion)

                    if label == "emotion_clause":
                        emotion_clauses.append(clause)
                    elif label == "cause_clause":
                        cause_clauses.append(clause)

                    classified_clauses.append({"clause": clause, "label": label})

                linked_emotions = []
                for emotion_clause in emotion_clauses:
                    linked_causes = [cause for cause in cause_spans if cause in utterance]
                    linked_emotions.append({"emotion_clause": emotion_clause, "caused_by": linked_causes})

                results[conv_id].append({
                    "turn": turn.get("turn"),
                    "speaker": turn.get("speaker"),
                    "utterance": utterance,
                    "emotion": emotion,
                    "clauses": classified_clauses,
                    "emotion_cause_mapping": linked_emotions
                })

    return results

# Example usage
if __name__ == "__main__":
    with open("dailydialog_train_clauses_aritra.json", "r", encoding="utf-8") as file:
        input_data = json.load(file)

    results = process_dataset(input_data)

    with open("results_final.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    print("Classification results saved to results_final.json")
