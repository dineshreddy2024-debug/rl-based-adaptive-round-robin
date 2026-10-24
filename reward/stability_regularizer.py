def stability_penalty(current_q, previous_q, coefficient=0.02):
    return coefficient * abs(current_q-previous_q)
