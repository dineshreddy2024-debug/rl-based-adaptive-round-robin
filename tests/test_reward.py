from reward.reward_function import MultiObjectiveReward
def test_reward():
    r=MultiObjectiveReward().compute({"waiting_time":1,"response_time":1,"context_switches":1,"cpu_utilization":90},4)
    assert isinstance(r,float)
