"""Truncate a modal basis to the first k modes (modes are ascending in energy)."""
import sys
import numpy as np
inp, out, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
z = np.load(inp, allow_pickle=True)
d = {n: z[n] for n in z.files}
d["modes"] = np.asarray(z["modes"])[:, :, :k]
d["eigvals"] = np.asarray(z["eigvals"])[:k]
np.savez_compressed(out, **d)
print(f"{out}: {d['modes'].shape[2]} modes")
