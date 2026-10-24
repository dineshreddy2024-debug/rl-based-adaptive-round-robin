class MultiObjectiveReward:
    def __init__(self, waiting_weight=1.0, response_weight=0.5, cs_weight=0.1,
                 utilization_weight=0.2, fairness_weight=0.2, quantum_penalty=0.02):
        self.ww=waiting_weight; self.rw=response_weight; self.csw=cs_weight
        self.uw=utilization_weight; self.fw=fairness_weight; self.qp=quantum_penalty
    def compute(self, info, quantum):
        # Negative cost: lower WT/RT/CS is better; higher utilization is better.
        return (-self.ww*info.get("waiting_time",0)
                -self.rw*info.get("response_time",0)
                -self.csw*info.get("context_switches",0)
                +self.uw*info.get("cpu_utilization",0)/100.0
                -self.qp*quantum)
