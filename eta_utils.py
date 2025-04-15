import numpy as np
import time

# Formatting functions


def time_format(secs):
    if np.abs(secs) < 60.:
        return f"{secs:.2f}s"
    else:
        return time_format_hhmmss(secs)


def time_format_hhmmss(secs):
    """
    Formats an integer secs into a HH:MM:SS format.

    :param secs:
    :return:
    """
    if type(secs) is not int:
        secs = int(secs)

    sign = np.sign(secs)
    secs = np.abs(secs)
    return f"{'' if sign >= 0 else '-'}{str(secs // 3600).zfill(2)}:{str((secs // 60) % 60).zfill(2)}:{str(secs % 60).zfill(2)}"


def iters_per_second_format(x):
    return f"{x:.2f} it/s" if x > 1. else f"{time_format(1. / x)}/iter"


# Classes


class ETAUpdate:
    """
    A struct to store (batched) iterations.
    """
    def __init__(self, time_taken, batch):
        self.time_taken = time_taken
        self.batch = batch

    def __iter__(self):
        for v in (self.time_taken, self.batch):
            yield v


class ETAUpdates:
    """
    An iterable that stores ETAUpdate classes.
    """
    raw_list: list[ETAUpdate]

    def __init__(self):
        self.raw_list = []

    def __iter__(self):
        for v in self.raw_list:
            yield v

    def append(self, eta_update):
        self.raw_list.append(eta_update)

    def times_taken(self):
        return np.array([u.time_taken for u in self.raw_list])

    def batches(self):
        return np.array([u.batch for u in self.raw_list])

    def numpy(self):
        return np.array((tuple(eta_update) for eta_update in self.raw_list))


class BaseETA:
    """
    The base class for ETA calculators. When making a custom ETA class, at least get_iters_per_second must be implemented.
    The function get_eta could also be worth re-implementing. The base version just calls get_iters_per_second, which works for most cases.
    """
    def __init__(self, total_iters):
        """

        :param total_iters: The total number of iterations
        """
        if int(total_iters) != total_iters:
            raise ValueError(f"total_iters could not be seen as the same as an integer (value given: {total_iters})")

        if total_iters <= 0:
            raise ValueError("total_iters must be strictly positive and an integer")

        self.current_iter = 0
        self.total_iters = total_iters

        self.updates = ETAUpdates()

        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.last_time_taken = 0.
        self.total_time_taken = 0.

        self.iters_per_second = float("nan")
        self.eta = float("nan")

    def update(self, batch=1):
        now = time.time()
        self.last_time_taken = now - self.last_update_time
        self.last_update_time = now
        self.total_time_taken = now - self.start_time
        self.current_iter += batch

        self.updates.append(ETAUpdate(time_taken=self.last_time_taken,
                                      batch=batch))

        self.iters_per_second = self.get_iters_per_second()
        self.eta = self.get_eta()

    def text(self, percent_digits=1, bar_length=10, bar_filled_char='=', bar_unfilled_char=' '):
        """
        Creates the text that's shown in the console through show_progress.
        Doesn't come with return characters or anything else to make the text "live".

        :param percent_digits: How many decimal places to show for the percentage.
        :param bar_length: How long the inner part of the bar is.
        :param bar_filled_char: The character to indicate the "filled" part of the bar.
        :param bar_unfilled_char: The character to indicate the "unfilled" part of the bar.
        :return: str
        """
        percent = self.current_iter / self.total_iters

        num_filled_ticks = int(bar_length * percent)
        num_unfilled_ticks = bar_length - num_filled_ticks

        loading_bar = '[' + (num_filled_ticks * bar_filled_char) + (num_unfilled_ticks * bar_unfilled_char) + ']'
        return f"{loading_bar} {self.current_iter}/{self.total_iters} ({('{' + f':.{percent_digits}%' + '}').format(percent)}) {iters_per_second_format(self.iters_per_second)} ETA {time_format(self.eta)}"

    def show_progress(self):
        """
        Prints text as a "live" console output. Must be called after update at every loop iteration.
        """
        print(f"\r{self.text()}", end="" if self.current_iter < self.total_iters else "\n")

    def get_iters_per_second(self):
        """
        Calculates iterations per second so far. After calling update, this is stored in the self.iters_per_second variable.

        Custom ETA classes must implement this, and optionally get_eta.

        :return: float
        """
        raise NotImplementedError("get_iters_per_second has not been implemented")

    def get_eta(self):
        """
        Calculates ETA so far. After calling update, this is stored in the self.eta variable.

        This function has the default behavior "iterations left / iterations per second", and could be re-implemented if desired.

        :return: float
        """
        return (self.total_iters - self.current_iter) / self.get_iters_per_second()


class SimpleAverageETA(BaseETA):
    """
    An ETA class that computes iterations per second with a simple arithmetic average.
    """
    def get_iters_per_second(self):
        return self.current_iter / self.total_time_taken


class ExponentiallyWeightedMovingAverageETA(BaseETA):
    """
    An ETA class that computes iterations per second with an Exponentially Weighted Moving Average.

    Given its discrete nature, it makes more sense to compute the average seconds per iteration first, and then inverting it.
    """
    def __init__(self, total_iters, alpha=0.05, iters_per_second_start_value=1.):
        """

        :param total_iters: The total number of iterations.
        :param alpha: Parameter between 0 and 1 that determines by how much to weigh recent data. The higher, the more recent iterations influence the estimator.
        :param iters_per_second_start_value: The starting value.
        """
        super().__init__(total_iters=total_iters)

        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be between 0 and 1")

        if iters_per_second_start_value <= 0:
            raise ValueError("iters_per_second_start_value must be positive")

        self.alpha = alpha
        self.iters_per_second_start_value = iters_per_second_start_value

    def get_iters_per_second(self):
        # clean the eta updates
        # treat a batched update as each element taking the same, average time
        times_taken_clean = []
        for update in self.updates:
            for _ in range(update.batch):
                times_taken_clean.append(update.time_taken / update.batch)

        I = len(times_taken_clean)

        # weighted average of time per iteration
        weights = [self.alpha * (1. - self.alpha) ** i for i in range(I, -1, -1)]

        average_time_per_iter = (sum([t * w for t, w in zip(
            [self.iters_per_second_start_value, *times_taken_clean],
            weights)]) / sum(weights))

        # return the reciprocal
        return 1. / average_time_per_iter
