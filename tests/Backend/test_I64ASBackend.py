import numpy as np
from Amber.Core.Backend import I64ASBackend


backend = I64ASBackend()



def test_encode_decode():
    out = backend.decode(backend.recode(backend.mul(backend.encode(999), backend.encode(-999))))
    assert np.allclose(out, -999 * 999)


if __name__ == '__main__':
    test_encode_decode()