
import numpy as np

def _is_valid_bio_transition(prev_label: str, curr_label: str) -> bool:
    """
    Check if a BIO transition is valid.

    Rules:
    - O can be followed by anything (O, B-*, I-*)
    - B-XXX can be followed by anything (I-XXX, B-*, O)
    - I-XXX can only be followed by I-XXX (same type) or anything non-I (B-*, O)

    This means I-XXX CANNOT follow I-YYY (different type) or B-YYY (different type).
    """
    if curr_label == "O":
        return True
    if curr_label.startswith("B-"):
        return True
    # curr_label starts with "I-"
    if prev_label == "O":
        return True
    if prev_label.startswith("B-") or prev_label.startswith("I-"):
        # Extract type and check if it matches
        curr_type = curr_label[2:]
        if prev_label.startswith("B-"):
            prev_type = prev_label[2:]
        else:  # prev_label.startswith("I-")
            prev_type = prev_label[2:]
        return curr_type == prev_type
    # prev_label is some other format, allow transition
    return True


def _constrained_argmax(
    logits: np.ndarray,
    id2label: dict[int, str],
) -> np.ndarray:
    """
    Perform argmax with BIO constraint enforcement.

    At each position, selects the highest-scoring label that maintains valid
    BIO transitions from the previous label, rather than patching invalid
    labels post-hoc.

    Args:
        logits: (n_tokens, n_labels) float array of raw model outputs
        id2label: mapping from label id to label string

    Returns:
        (n_tokens,) int64 array of selected label ids
    """
    n_tokens = logits.shape[0]
    o_id = next((k for k, v in id2label.items() if v == "O"), None)
    if o_id is None:
        raise ValueError("id2label must contain an 'O' label")

    out = np.empty(n_tokens, dtype=np.int64)
    prev_label = "O"

    for i in range(n_tokens):
        best_valid: int | None = None
        for lid in np.argsort(logits[i])[::-1]:
            curr_label = id2label.get(int(lid), "O")
            if _is_valid_bio_transition(prev_label, curr_label):
                best_valid = int(lid)
                break
        if best_valid is None:
            best_valid = o_id
        out[i] = best_valid
        prev_label = id2label[int(best_valid)]

    return out
