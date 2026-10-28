## Issue #147: parallel workers

Optuna 5.0.0 QMCSampler's study-attribute counter is a non-atomic read/write.
Use the storage-assigned trial number for Sobol indexing instead, and refresh
warm-up progress from shared completed trials at each before_trial call.
The override depends on Optuna 5's private _find_sample_id method (dependency
is pinned below 6); the multiprocessing test checks the exact Sobol indices.
`constant_liar=False` is the requested public default, although Optuna 5's
own default has changed to True. Preflight has no built-in file writes;
custom hooks must isolate their output paths across processes.
