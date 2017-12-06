import numpy as np
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import pdist, squareform


def permute_data(data, within_class_groups, pruned_groups):
    """Permute data based on grouped information.

    For each group in within_class_groups, permute the rows with respect to the
    labels given in the PClass column. For each group in pruned_groups, permute
    rows without respect to PClass.

    :param data: the dataset
    :param within_class_groups: groups to be permuted w.r.t PClass
    :param pruned_groups: groups to be permuted w/o.r.t PClass
    :return: permuted dataset
    """
    newdata = data.copy()
    classes = newdata.PClass.unique()
    for group in within_class_groups:
        for c in classes:
            permutation = newdata.loc[data.PClass == c, group].sample(frac=1)
            permutation.index = newdata.index[newdata.PClass == c]
            newdata.loc[newdata.PClass == c, group] = permutation
    for group in pruned_groups:
        newdata[group] = newdata[group].sample(frac=1).reset_index(drop=True)
    return newdata


def fidelity(data, model, within_class_groups, pruned_groups):
    """Compute fidelity.

    :param data: the dataset
    :param model: trained classification model
    :param within_class_groups: groups to be permuted w.r.t PClass
    :param pruned_groups: groups to be permuted w/o.r.t PClass
    :return: fidelity score
    """
    newdata = permute_data(data, within_class_groups, pruned_groups)
    newpred = model.predict(newdata.drop('PClass', axis=1))
    fid = accuracy_score(data.PClass, newpred)

    return fid


def total_variation_distance(u_values, v_values):
    """Compute the total variation distance between two 1D distributions.

    :param u_values: probability distribution
    :param v_values: probability distrbution
    :return: total variation distance between u_values and v_values
    """
    dist = sum([abs(p-q) for (p, q) in zip(u_values, v_values)])
    return dist


def distance_correlation(pks, qks, metric=total_variation_distance):
    """Compute distance correlation between two sets of probabilty distributions.

    See also: https://en.wikipedia.org/wiki/Distance_correlation

    :param pks: list of probability distributions
    :param qks: list of probability distributions
    :param metric: used distance metric
    :return: distance correlation
    """
    pks_distmat = squareform(pdist(pks, metric=metric))
    qks_distmat = squareform(pdist(qks, metric=metric))

    # compute row, column and grand means
    pks_rmeans = np.mean(pks_distmat, axis=0)
    qks_rmeans = np.mean(qks_distmat, axis=0)
    pks_cmeans = np.mean(pks_distmat, axis=1)
    qks_cmeans = np.mean(qks_distmat, axis=1)
    pks_gmean = np.mean(pks_distmat)
    qks_gmean = np.mean(qks_distmat)

    # doubly centered distances
    pks_dcds = [pks_distmat[i,j] - pks_rmeans[i] - pks_cmeans[j] + pks_gmean
                for i in range(len(pks))
                for j in range(len(pks))]
    qks_dcds = [qks_distmat[i, j] - qks_rmeans[i] - qks_cmeans[j] + qks_gmean
                for i in range(len(qks))
                for j in range(len(qks))]

    # covariance and standard deviations
    cov = np.sqrt(np.sum([pks_dcds[i] * qks_dcds[i] for i in range(len(pks))]) / (len(pks)**2))

    pks_sd = np.sqrt(np.sum([p**2 for p in pks_dcds]) / (len(pks)**2))
    qks_sd = np.sqrt(np.sum([q**2 for q in qks_dcds]) / (len(qks)**2))

    # correlation
    cor = cov / np.sqrt(pks_sd*qks_sd)

    return cor


def class_probability_ranking(data, model, within_class_groups, pruned_groups):
    """Compute class probability ranking using the given grouping.

    NOTE: This implementation differs from the original authors' approach.
    While the original approach is limited to binary classification, in this
    implementation it is extended to compare non-binary classifications.

    The class probability ranking computes the correlation between the original
    prediction probabilities and the prediction probabilities created by
    permuting the data based on the given grouping.

    :param data: the dataset
    :param model: trained classification model
    :param within_class_groups: groups to be permuted w.r.t PClass
    :param pruned_groups: groups to be permuted w/o.r.t PClass
    :return: correlation between original prediction probabilities and
    predicition probabilities on the permuted data
    """
    if not hasattr(model, 'predict_proba'):
        raise NotImplementedError("The provided model does not support"
                                  "predicting probabilities.")

    newdata = permute_data(data, within_class_groups, pruned_groups)
    pred = model.predict_proba(data.drop('PClass', axis=1))
    newpred = model.predict_proba(newdata.drop('PClass', axis=1))

    cor = distance_correlation(pred, newpred)

    return cor
