from workloads.generator import WORKLOAD_TYPES,generate_workload
def test_workloads():
    for k in WORKLOAD_TYPES:
        assert len(generate_workload(5,k,1))==5
