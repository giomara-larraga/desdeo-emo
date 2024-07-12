import numpy as np

# import plotly.graph_objects as go

from desdeo_emo.utilities.ReferenceVectors import ReferenceVectors
from pygmo import fast_non_dominated_sorting as nds


def generate_composite_front(*fronts):

    _fronts = np.vstack(fronts)

    cf = _fronts[nds(_fronts)[0][0]]

    return cf


def generate_composite_front_with_identity(*fronts):
    # This is currently working for two fronts
    # First two fronts should be the individual fronts from each algorithm to be compared
    # This function counts the number of solutions provided to the composite front by each algorithm to be compared.

    first_front = np.shape(fronts[0])
    second_front = np.shape(fronts[1])

    _fronts = np.vstack(fronts)
    # print(nds(_fronts)[0][0])

    temp = nds(_fronts)[0][0]
    first_nds = temp[temp < first_front[0] - 1]
    second_nds = temp[temp > first_front[0] - 1]

    # Following lines are needed since composite front is keeping all the solutions from the very beginning.
    # I am finding always the newly added nondominated solutions after each iteration by each algorithm
    remaining_fronts = (first_front[0]) + (second_front[0])
    remaining_nds = temp[temp > remaining_fronts]

    # print(remaining_nds)

    first = first_nds.shape[0]
    second = second_nds.shape[0]
    second -= remaining_nds.shape[0]

    cf = _fronts[temp]

    return first, second, cf


def translate_front(front, ideal):
    translated_front = np.subtract(front, ideal)
    return translated_front


def normalize_front(front, translated_front):
    translated_norm = np.linalg.norm(translated_front, axis=1)
    translated_norm = np.repeat(translated_norm, len(translated_front[0, :])).reshape(
        len(front), len(front[0, :])
    )

    translated_norm[translated_norm == 0] = np.finfo(float).eps
    normalized_front = np.divide(translated_front, translated_norm)
    return normalized_front


def assign_vectors(front, vectors: ReferenceVectors):
    cosine = np.dot(front, np.transpose(vectors.values))
    if cosine[np.where(cosine > 1)].size:
        cosine[np.where(cosine > 1)] = 1
    if cosine[np.where(cosine < 0)].size:
        cosine[np.where(cosine < 0)] = 0

    theta = np.arccos(cosine)  # check this theta later, if needed or not
    assigned_vectors = np.argmax(cosine, axis=1)

    return assigned_vectors, theta


class baseADM:
    def __init__(self, composite_front, vectors: ReferenceVectors):

        self.composite_front = composite_front
        self.vectors = vectors
        self.ideal_point = composite_front.min(axis=0)
        self.translated_front = translate_front(self.composite_front, self.ideal_point)
        self.normalized_front = normalize_front(
            self.composite_front, self.translated_front
        )
        self.assigned_vectors, self.theta = assign_vectors(
            self.normalized_front, self.vectors
        )
