import time
from IPython.display import clear_output


class Progressbar:
    def __init__(self, experiments_n):
        self.experiments_n = experiments_n
        self.disabled = False
        self.bar_length = 20
        self.text = ""
        self.time = time.time()

    def update_experiments_n(self, experiments_n):
        self.experiments_n = experiments_n

    def experiment(self, experiment_name):
        self.experiment_name = experiment_name
        self.text += "Experiment: " + experiment_name + "\n"

    def loaded_experiment(self, algorithm_name):
        self.text += '\tLoaded experiment results: {}\n'.format(algorithm_name)
        print(self.text)

    def algorithm(self, algorithm_name):
        self.algorithm_name = algorithm_name
        self.algorithm_start = time.time()
        self.update_progress(-1)

    def disable(self):
        self.disabled = True

    def enable(self):
        self.disabled = False

    def gen_samples_start(self):
        self.current_algorithm = self.algorithm_name
        self.algorithm("Generate samples")

    def gen_samples_end(self):
        self.done_algorithm()
        self.algorithm(self.current_algorithm)

    def done_algorithm(self):
        self.text += self.get_current_text(1) + '\n'

    def get_current_text(self, progress):
        block = int(round(self.bar_length * progress))
        if progress == 0:
            hours, minutes, seconds = 0.0, 0.0, 0.0
            total_hours, total_minutes, total_seconds = 0.0, 0.0, 0.0
        else:
            total_time = (1 / progress) * (time.time() - self.algorithm_start)
            remaining_time = total_time - (time.time() - self.algorithm_start)
            total_hours, total_minutes, total_seconds = total_time // 3600, (total_time % 3600) // 60, (
                    (total_time % 3600) % 60)
            if progress == 1:
                hours, minutes, seconds = 0.0, 0.0, 0.0
            else:
                hours, minutes, seconds = remaining_time // 3600, (remaining_time % 3600) // 60, (
                        (remaining_time % 3600) % 60)
        return f"\tBenchmark: " + self.algorithm_name + ", Progress: [{0}] {1:.1f}%\n\tEstimated total time: {" \
                                                        "2:02.0f}:{3:02.0f}:{4:02.0f} \t Estimated time left: {" \
                                                        "5:02.0f}:{6:02.0f}:{7:02.0f}\n".format(
            "#" * block + "-" * (self.bar_length - block), progress * 100, total_hours, total_minutes, total_seconds,
            hours, minutes, seconds)

    def update_progress(self, i, k=0):
        if k == 0: k = self.experiments_n
        progress = (i + 1) / k
        if not self.disabled:
            clear_output(wait=True)
            print(self.text + self.get_current_text(progress))
