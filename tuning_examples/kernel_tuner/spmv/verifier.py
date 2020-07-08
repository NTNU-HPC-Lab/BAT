import numpy

def verify_multiple_answers(answer, gpu_result, atol=None):
    def _ravel(a):
        if hasattr(a, 'ravel') and len(a.shape) > 1:
            return a.ravel()
        return a

    def _flatten(a):
        if hasattr(a, 'flatten'):
            return a.flatten()
        return a

    def check_similarity(ans, gpu_res):
        for i in range(0, len(ans)): 
            expected = ans[i]
            if expected is not None:
                result = _ravel(gpu_res[i])
                expected = _flatten(expected)
                return numpy.allclose(expected, result, atol=0.02)
        return False # This will fail if every value is None, but no verification is needed in that case

    passing = False
    if isinstance(answer[0], list):
        for i in range(0, len(answer)):
            res = check_similarity(answer[i], gpu_result)
            if res is True:
                passing = True
        return passing
    else:
        res = check_similarity(answer, gpu_result)
        if res is False:
            print('Answers does not match')
        return res
                
    return True