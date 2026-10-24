class FallbackController:
    def __init__(self, default_quantum=4, confidence_threshold=0.5):
        self.default_quantum=default_quantum
        self.threshold=confidence_threshold
    def choose(self, action, confidence, safe_action=1):
        return action if confidence >= self.threshold else safe_action
