import query_complexity as qc

sol = qc.analytic_or_solution(2)
td = qc.build_phase_transducer(sol)

print("OR_2 analytic phase solution")
print("objective:", sol.objective_value)
print("public/private dim:", td.public_dim, td.private_dim)
print("gram/map/unitarity errors:", td.gram_error, td.map_error, td.unitarity_error)
print("U:")
print(td.U)
qc.show_catalysts(sol)
