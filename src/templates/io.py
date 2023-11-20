import inspect
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def determine_input_type(fcn, arg_names=['X', 'y']):

    # Get parameters
    sgn = inspect.signature(fcn)
    inp = list (set(sgn.parameters.keys()).intersection(arg_names) )

    # Get types
    typ = {}
    for i_ in inp:
        i_sgn = sgn.parameters[i_].annotation
        # Check if list
        if hasattr(i_sgn, '__args__'): # This is a tuple
            typ[i_] = list( i_sgn.__args__ )
        else:
            typ[i_] = [i_sgn]
    return typ


def get_transformer_outputs(obj):
    # instantiate new object
    n_obj = obj.__class__()

    # Get object signature
    sgn = determine_input_type(n_obj.fit_transform, arg_names=['X', 'y'])
    try:  # with digits

        # --- Prep inputs
        # Init test arrays
        X, y = np.random.random([10, 4]), np.random.randint([10], size=10)
        if pd.DataFrame in sgn['X']:
            X = pd.DataFrame(data=X, columns=[f'Col_{i_}' for i_ in range(4)])
        if pd.DataFrame in sgn['y']:
            y = pd.DataFrame(data=y, columns=['label'])

            # Make output
        out = n_obj.fit_transform(X, y)

    except:  # with strings

        # instantiate new object
        n_obj = obj.__class__()

        try:
            # Init test arrays
            X, y = ['This is a test', 'This only determines the signature of the transformer', 'Use with caution'], [1,
                                                                                                                     0,
                                                                                                                     1]
            if pd.DataFrame in sgn['X']:
                X = pd.DataFrame(data=X, columns=['features'])
            if pd.DataFrame in sgn['y']:
                y = pd.DataFrame(data=y, columns=['label'])

                # Make output
            out = n_obj.fit_transform(X, y)
        except:
            raise TypeError('Transformer does not work for digits nor for strings')

    # Determine N
    n_out = 1
    if isinstance(out, tuple):
        n_out = len(out)
    return n_out


