from numpy import random
from eta_utils import SimpleAverageETA, ExponentiallyWeightedMovingAverageETA, time_format

from time import sleep

rng = random.default_rng(seed=100)

N = 1000
mean_sleep = 0.01
exponentially_weighted = False

if exponentially_weighted:
    eta = ExponentiallyWeightedMovingAverageETA(total_iters=N)
else:
    eta = SimpleAverageETA(total_iters=N)


for n in range(1, N+1):
    sleep(rng.exponential(mean_sleep))

    eta.update()

    eta.show_progress()

print(f"Done, took {time_format(eta.total_time_taken)}")
