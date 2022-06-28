from reader import core


class OpenTunerLikeAutotuner:
    def pickNewConfig(self, result):
        tuning_config = {
            "BLOCK_SIZE": 256,
            "TEXTURE_MEMORY": 0,
            "WORK_PER_THREAD": 1
        }
        return tuning_config

    def run(self, run_cmd):
        tuning_config = self.pickNewConfig(0)
        budget = 100
        for i in range(budget):
            result = run_cmd(tuning_config)
            tuning_config = self.pickNewConfig(result)


def coreOpenTuner(tuning_config):
    benchmark_config = {
        "PRECISION": 32
    }
    return core('MD-CAFF.json', benchmark_config, tuning_config)


class GenericAutotuner:
    def pickNewConfig(self, result):
        tuning_config = {
            "BLOCK_SIZE": 256,
            "TEXTURE_MEMORY": 0,
            "WORK_PER_THREAD": 1
        }
        return tuning_config

    def run(self, result):
        return self.pickNewConfig(result)


def main():
    budget = 1
    benchmark_config = {
        "PRECISION": 32
    }
    typeOfAutotuner = "Direct"

    if typeOfAutotuner == "Direct":
        at = GenericAutotuner()
        tuning_config = at.run(0)
        for i in range(budget):
            result = core('MD-CAFF.json', benchmark_config, tuning_config)
            tuning_config = at.run(result)
    else:
        openTuner = OpenTunerLikeAutotuner()
        openTuner.run(coreOpenTuner)


if __name__ == "__main__":
    main()
