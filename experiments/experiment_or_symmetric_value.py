import query_complexity as qc

import mosek

n = 60

sol = qc.solve_symmetric_value(n, layer_values = qc.OR_layers(n), verbose = True)
