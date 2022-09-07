import cupy as cp

TESTING = True


def reduce_correctness(args_before, args_after, tuning_config, launch_config):
    accuracy = 3
    left = round(sum(args_before[0]).item(), accuracy)
    right = round(sum(args_after[1][:launch_config["GRID_SIZE_X"]]).item(), accuracy)
    if left != right:
        print("Sum: {} != {}".format(left, right))
    else:
        print("Passed", tuning_config)


def generic_correctness(args_before, args_after, tuning_config, launch_config):
    if not TESTING:
        print("Correctness")
        print(tuning_config)
        print(args_before)
        print(args_after)


def builtin_vectors_correctness(args_before, args_after, tuning_config, launch_config):
    left = args_after[2][0].item()[0]
    right = args_before[0][0].item()[0] + args_before[1][0].item()[0]
    if left == 0:
        print("Failed to initialize", args_after[3])
    elif left != right:
        print("Did not pass", left, "!=", right, args_after[3], tuning_config)
        exit(1)
    else:
        print("Passed", tuning_config)


def md5hash_correctness(args_before, args_after, tuning_config, launch_config):
    key = args_after[8]
    reference = cp.asarray([9, 5, 7, 9, 8, 9, 9, 0], dtype=cp.byte)
    if (key == reference).all():
        print("Passed", tuning_config)
    else:
        print("Failed correctness:", key, reference)


correctness_funcs = {
    "sum_kernel": builtin_vectors_correctness,
    "compute_lj_force": generic_correctness,
    "reduce": reduce_correctness,
    "FindKeyWithDigest_Kernel": md5hash_correctness,
    "FFT512_device": generic_correctness,
    "triad": generic_correctness
}
