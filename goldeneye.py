from math import ceil, sqrt

from sklearn.model_selection import train_test_split

from goodness import fidelity


def singletonize(groups):
    """Singletonize each element of the contained lists.

    Example: [[1,2],[3],[4,5,6]] -> [[1],[2],[3],[4],[5],[6]]
    :param groups: list of lists to be singletonized
    :return: singletonized lists
    """
    singles = []
    for group in groups:
        singles.extend([[el] for el in group])
    return singles


def extract_el_from_group(group, el):
    """Extract an element group from a group.

    :param group: list
    :param el: element to be extracted
    :return: group without the extracted element, the extracted element
    """
    extracted_group = [x for x in group if x != el]
    return [extracted_group] + [[el]]


def remove_el_from_group(group, el):
    """Remove an element from a group.

    :param group: list
    :param el: element to be removed
    :return: new group without the element
    """
    return [x for x in group if x != el]


def grouping(data, model, delta, classname, goodness_fn):
    """Find the optinal grouping of attributes for a dataset.

    This function iteratively finds the optimal grouping of attributes for a
    dataset using a given classifier.

    :param data: the dataset
    :param model: trained classification model
    :param delta: sensitivity parameter
    :param classname: name of the column in the dataset with the class labels
    :param goodness_fn: function used to investigate the effect of randomising
    attributes in the dataset
    :return: detected optimal groups, naive goodness, inflated data
    """
    attributes = [col for col in data.columns
                  if col not in [classname, 'PClass']]

    # compute naive bayes goodness using an inflated dataset,
    # to ensure precision
    p = 0.5
    sgm = delta / 5
    n = ceil(p * (1 - p) / (sgm ** 2))
    inflated_data = data.sample(n, replace=True)

    nb_goodness = goodness_fn(inflated_data, model,
                              singletonize(attributes), [])

    # inflate the dataset to ensure that the desired variance level can be
    # reached
    p = max(0.5, nb_goodness)
    n = max(1000, ceil((p * (1 - p)) / (sgm ** 2)))
    inflated_data = data.sample(n, replace=True)

    # --------------------------------------------------

    detected_groups = []  # accumulated attribute groups
    current_group = attributes  # currently tested group
    removed_attrs = []  # removed attributes
    Delta = nb_goodness + delta  # grouping threshold

    # group the attributes
    while current_group or removed_attrs:
        current_goodness = \
            goodness_fn(inflated_data, model,
                        [current_group] + singletonize(detected_groups), [])
        if not removed_attrs and current_goodness < Delta:
            # already below Delta before removing any attributes,
            # so assign remaining attributes to singleton groups
            detected_groups.extend(singletonize([current_group]))
            current_group = []
            removed_attrs = []
        else:
            # find the attribute that decreases the goodness the least
            goodnesses = [
                goodness_fn(inflated_data, model,
                            extract_el_from_group(current_group, attr) +
                            singletonize(detected_groups) + [removed_attrs],
                            [])
                for attr in current_group]

            max_goodness = max(goodnesses)
            if len(current_group) == 1 or max_goodness < Delta:
                # if the goodness drops below Delta, add the group of attributes
                # to the result and look for the next group of attributes
                detected_groups.append(current_group)
                current_group = removed_attrs
                removed_attrs = []
            else:
                # if the goodness stays above Delta,
                # continue with the current group (minus the removed attribute)
                attr_idx = goodnesses.index(max_goodness)
                removed_attrs.append(current_group[attr_idx])
                current_group = current_group[:attr_idx] + \
                                current_group[attr_idx + 1:]

    return detected_groups, nb_goodness, inflated_data


def prune_singletons(data, model, delta, groups, goodness_fn):
    """Prune singleton attributes.

    This function iteratively prunes those singletons from a given grouping
    that do not affect the fidelity more than the given sensitivity.

    :param data: the dataset
    :param model: trained classification model
    :param delta: sensitivity parameter
    :param groups: detected optimal grouping
    :param goodness_fn: function used to investigate the effect of randomising
    attributes in the dataset
    :return: grouping with the pruned attributes, pruned attributes
    """
    groups_pruned = groups[:]
    pruned_singletons = []

    # original goodness
    Delta = goodness_fn(data, model, groups, []) - delta

    # receive singleton attributes
    singletons = [group for group in groups if len(group) == 1]

    # prune singletons
    while singletons:
        # find the singleton which decreases the goodness the least
        goodnesses = [
            goodness_fn(data, model,
                        remove_el_from_group(groups, singleton), [singleton])
            for singleton in singletons]

        max_goodness = max(goodnesses)

        if max_goodness >= Delta:
            # prune the singleton and continue
            single_idx = goodnesses.index(max_goodness)
            pruned_singletons.extend(singletons[single_idx])
            groups_pruned.remove(singletons[single_idx])
            singletons = singletons[:single_idx] + singletons[single_idx + 1:]
        else:
            singletons = []

    return groups_pruned, pruned_singletons


def goldeneye(data, model, delta=None, classname='Class', goodness_fn=fidelity):
    """Detect optimal groups in a dataset.

    :param data: the dataset
    :param model: untrained classification model
    :param delta: sensitivity parameter
    :param classname: name of the column in the dataset with the class labels
    :param goodness_fn: function used to investigate the effect of randomising
    attributes in the dataset
    :return: detected groups, goodness with the detected groups, pruned groups,
    goodness with the pruned groups, naive goodness, original accuracy, final
    accuracy, delta
    """
    if delta is None:
        delta = 1 / sqrt(data.shape[0])

    if classname not in data.columns:
        raise ValueError("classname not found in dataset")

    # learn model
    X_train, X_test, y_train, y_test = \
        train_test_split(data.drop(classname, axis=1), data[classname],
                         test_size=0.5, stratify=data.Label)
    model.fit(X_train, y_train)

    data = X_test.reset_index(drop=True)

    data['PClass'] = model.predict(data)

    # find optimal grouping of the dataset
    groups, nb_goodness, inflated_data = \
        grouping(data, model, delta, classname, goodness_fn)
    goodness = goodness_fn(inflated_data, model, groups, [])

    # prune singletons
    groups_pruned, pruned_singletons = \
        prune_singletons(inflated_data, model, delta, groups, goodness_fn)
    goodness_pruned = \
        goodness_fn(inflated_data, model, groups_pruned, pruned_singletons)

    acc_original = fidelity(inflated_data, model, [], [])
    acc_final = fidelity(inflated_data, model, groups_pruned, pruned_singletons)

    return groups, goodness, groups_pruned, goodness_pruned, nb_goodness, \
        acc_original, acc_final, delta
