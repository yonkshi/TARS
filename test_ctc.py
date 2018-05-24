import numpy as np

SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space

def convert_inputs_to_ctc_format(target_text):
    print(target_text)

    # Transform in 3D arra

    # Get only the words between [a-z] and replace period for none
    original = ' '.join(target_text.strip().lower().split(' ')).replace('.', '').replace('?', '').replace(',', '').replace("'", '').replace('!', '').replace('-', '')
    print(original)
    targets = original.replace(' ', '  ')
    targets = targets.split(' ')

    # Adding blank label
    targets = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in targets])

    # Transform char into index
    targets = np.asarray([SPACE_INDEX if x == SPACE_TOKEN else ord(x) - FIRST_INDEX
                          for x in targets])

    # Creating sparse representation to feed the placeholder
    train_targets = sparse_tuple_from([targets])

    return train_targets


def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


txt = 'hello world'
convert_inputs_to_ctc_format(txt)