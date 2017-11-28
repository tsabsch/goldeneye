from sklearn.metrics import accuracy_score


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
