# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

# TODO: Write some benchmarks for parcels


class ExampleTimeSuite:
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
    """

    def setup(self):
        self.d = {}
        for x in range(500):
            self.d[x] = None

    def time_keys(self):
        for _ in self.d.keys():
            pass

    def time_values(self):
        for _ in self.d.values():
            pass

    def time_range(self):
        d = self.d
        for key in range(500):
            _ = d[key]


class ExampleMemSuite:
    def mem_list(self):
        return [0] * 256
