# ms_eta_utils
 
A simple library to quickly have an ETA meter, for long-running loops.

For example, the loop:
```
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

Requires `numpy` to run.