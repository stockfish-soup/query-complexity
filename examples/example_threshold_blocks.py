import query_complexity as qc

n, k = 4, 2
blocks = qc.threshold_eta_blocks(n, k)
print(f"THRESHOLD^{k}_{n} analytic Schrijver block factors")
for r, data in blocks.items():
    print("r =", r)
    print("labels:", data["labels"])
    print("eta:", data["eta"])
    print("B:")
    print(data["B"])
