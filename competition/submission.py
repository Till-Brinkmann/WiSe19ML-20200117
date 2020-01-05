# -*- coding: utf-8 -*-
import numpy as np
# TODO: You can add additional imports - if you do so, please list all additional packages (including their version) in a separate file called "REQUIREMENTS.txt".
# However, all of your python code must be in this file - you can NOT submit other/more python files than this one!

# ***************************************************************************
# ATTENTION: USE 4 BLANKS FOR ENCODING A TAB - DO NOT CHANGE THIS ENCODING! *
# ***************************************************************************

# *************************************************************************
# CAUTION: DO NOT CHANGE THE NAME OR SIGNATURE OF THE FOLLOWING FUNCTION! *
# *************************************************************************
def my_submission(X_train, y_train, X_test):
    """
    Function containing your model-(pipeline) submission to the "IntroML-2019 competition".

    Parameters
    ----------
    X_train : `numpy.ndarray`
        Training samples, shape `(n_samples_in_X_train, n_features)`.
    y_train : `numpy.ndarray`
        Class labels (`int`) of training samples, shape `(n_samples_in_X_train, )`.
    X_test : `numpy.ndarray`
        Testing samples, shape `(n_samples_in_X_test, n_features)`.

    Returns
    -------
    `numpy.ndarray`
        One dimensional array of predicted class labels (`int`). Must be of size `(n_samples_in_X_test,)`. Make sure that the predicted class labels are integers!
    """
    # TODO: Create and fit your model (including preprocessing and feature extraction)
    # Put your model-(pipeline) here!

    # TODO: Compute predictions on X_test by using your trained model-(pipeline)
    pred_test = np.zeros(X_test.shape[0])  # TODO: Remove dummy prediction - Classify everything as class 0

    return pred_test	# Do not forget to return your predictions ;)
