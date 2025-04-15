# ms_eta_utils
 
A simple library to quickly have an ETA meter, for long-running loops.

For example, the loop:
```
N = 1000
eta = SimpleAverageETA(total_iters=N)

for n in range(N):
    <do calculations>

    eta.update()

    eta.show_progress()
```

Would show this in the console:
```commandline
[=         ] 159/1000 (15.9%) 88.65 it/s ETA 9.49s
```

Comes with `SimpleAverageETA` and `ExponentiallyWeightedMovingAverageETA` classes, with the base class `BaseETA`.
Batched iterations are implemented, and the batch size of one iteration can be specified with the `batch` parameter in `eta.update()`.

Since this is very simple code, you can just copy the `eta_utils.py` file right into your project.

Requires `numpy` to run.