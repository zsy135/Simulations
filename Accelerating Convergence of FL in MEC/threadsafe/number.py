import numbers

from lock import RWLock
import numbers


class ValueRW:
    def __init__(self, v_: numbers.Number):
        if not isinstance(v_, numbers.Number):
            raise TypeError("is not a number!!!!!!!")
        self._v = v_
        self.lock = RWLock()

    @property
    def value(self):
        _v = None
        self.lock.reader_acquire()
        _v = self._v
        self.lock.reader_release()
        return _v

    @value.setter
    def value(self, v_: numbers.Number):
        self.lock.writer_acquire()
        self._v = v_
        self.lock.writer_release()


# test
if __name__ == "__main__":

    try:
        d = NumberRW("123")
    except TypeError:
        pass
    else:
        assert False

    d = NumberRW(1)
    assert 1 == d.value
    d.value = 2
    assert d.value == 2