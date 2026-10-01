# Tensor worker protocol, version 1

`tensor-worker` reads UTF-8 JSON lines from stdin. Each nonempty line is a
calculation. Close stdin after sending a request when using one worker per job.
Stdout contains JSON lines only; clients must handle lines split across pipe
reads. A recoverable input error does not prevent later requests in the same
process. Cancellation exits the process after the active operation yields;
clients can force termination and start a new process when necessary.

```json
{
  "version": 1,
  "id": "calculation-1",
  "coordinates": "theta, phi",
  "scalars": "L",
  "functions": "",
  "metric": "diag(L^2, L^2*sin(theta)^2)",
  "outputs": ["ricci_scalar", "kretschmann"]
}
```

Output keys: `inverse_metric`, `christoffel`, `riemann`, `ricci`,
`ricci_scalar`, `kretschmann`. Omitting `outputs` selects all outputs; an empty
selection is rejected. Only requested outputs are returned; dependencies may
still appear as progress stages.

Every event has `version: 1`, the original string `id`, and `type`:

- `progress`: `stage` output key and a human-readable `message`.
- `error`: human-readable `message`.
- `result`: a nested `result` object containing `coordinates`,
  `elapsed_seconds`, `convention`, `tensors`, `scalars`, and `warnings`.

A tensor contains `key`, `name`, `symbol` (LaTeX), `variance` (`u`/`l`), `rank`,
`shape`, and `components`. Each component contains zero-based `indices`,
canonical `expression`, display `latex`, and `zero_status`. Only proven zeros
are omitted; `undetermined` means a nonzero printed expression whose zero
status is not proven. A scalar contains `key`, `name`, `symbol`, `expression`,
`latex`, and `zero_status`. Display strings do not define algebraic equality.

The two-sphere example returns `R = -2/L^2` and `K = 4/L^4` under the documented
curvature convention. Errors use the request id when it can be decoded; malformed
JSON returns an empty id. Inputs are limited to 1 MiB per line.

Optional request field `riemann_sign` is +1 (default, original derivative-nu-first convention) or -1 (the opposite convention). Other values are rejected. The result convention string records the full definition. The native UI converts LaTeX and applies the selected overall metric sign before submitting the metric; the worker continues to accept the restricted plain mathematical grammar.
