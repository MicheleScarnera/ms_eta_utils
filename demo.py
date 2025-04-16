from numpy import random
from eta_utils import SimpleAverageETA, ExponentiallyWeightedMovingAverageETA, time_format

from time import sleep

rng = random.default_rng(seed=100)

N = 1000
mean_sleep = 0.01
mean_batch_size = 5
exponentially_weighted = True

if exponentially_weighted:
    eta = ExponentiallyWeightedMovingAverageETA(total_iters=N)
else:
    eta = SimpleAverageETA(total_iters=N)

n = 0
while n < N:
    k = min(rng.geometric(1. / mean_batch_size), N - n)

    sleep(sum(rng.exponential(mean_sleep, size=k)))

    eta.update(batch=k)

    eta.show_progress()

    n += k

print(f"Done, took {time_format(eta.total_time_taken)}")
