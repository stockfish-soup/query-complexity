"""Small analytic OR example; no external solver or license required."""
import query_complexity as qc

sol = qc.analytic_or_solution(3)
transducer = qc.build_adversary_transducer(sol)
print("objective:", sol.objective_value)
print("private dimension:", transducer.private_dim)
print("map error:", transducer.map_error)
