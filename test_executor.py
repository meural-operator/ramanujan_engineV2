from clients.engine_bridge.executor import RamanujanExecutor

ex = RamanujanExecutor()
mock_wu = {
    "id": "test-001",
    "v2_bound_id": "bound_a_test_b_test",
    "constant_name": "euler-mascheroni",
    "a_deg": 2,
    "b_deg": 2,
    "a_coef_range": [[-3, -1], [-3, -1], [-3, -1]],
    "b_coef_range": [[-3, -1], [-3, -1], [-3, -1]]
}
hits = ex.execute_work_unit(mock_wu)
print(f"Hits found: {len(hits)}")
