__author__ = 'thiagocastroferreira'

def count(eqs = [], neqs = [], dataset = []):
    if len(eqs) > 0:
        dataset = count_eqs(eqs, dataset)

    if len(neqs) > 0:
        dataset = count_neqs(neqs, dataset)
    return dataset

def count_eqs(eqs = [], dataset = []):
    head, tail = eqs[0], eqs[1:]
    if len(tail) == 0:
        return filter(lambda x: x[head['key']] == head['value'], dataset)
    else:
        return count_eqs(tail, filter(lambda x: x[head['key']] == head['value'], dataset))

def count_neqs(neqs = [], dataset = []):
    head, tail = neqs[0], neqs[1:]
    if len(tail) == 0:
        return filter(lambda x: x[head['key']] != head['value'], dataset)
    else:
        return count_neqs(tail, filter(lambda x: x[head['key']] != head['value'], dataset))