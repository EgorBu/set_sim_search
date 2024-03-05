import logging
from multiprocessing import Process, Queue
import os
from typing import Callable, Iterable, Sequence, Set, Tuple

from numba import float64, njit, prange
import numpy as np
import psutil
import sharedmem
from tqdm import tqdm


@njit(nogil=True, fastmath=True)
def is_close(a, b, rel_tol=1e-05, abs_tol=1e-08):
    return abs(a - b) <= (abs_tol + rel_tol * abs(b))


@njit(nogil=True)
def _intersect(a: np.ndarray, b: np.ndarray) -> int:
    """
    Calculate size of intersection for sorted feature vectors
    :param a: first sorted feature vector
    :param b: second sorted feature vector
    :return: size of intersection
    """
    intersect_size = 0
    ptr_a = 0
    ptr_b = 0
    len_a = len(a)
    len_b = len(b)
    while ptr_a < len_a and ptr_b < len_b:
        while ptr_a < len_a and a[ptr_a] < b[ptr_b]:
            ptr_a += 1
        if ptr_a >= len_a:
            return intersect_size
        while ptr_b < len_b and b[ptr_b] < a[ptr_a]:
            ptr_b += 1
        while ptr_a < len_a and ptr_b < len_b and a[ptr_a] == b[ptr_b]:
            ptr_a += 1
            ptr_b += 1
            intersect_size += 1
    return intersect_size


@njit(nogil=True)
def _jaccard(query_feat: np.ndarray, candidate_feat: np.ndarray) -> float:
    """
    Jaccard similarity for sorted vectors
    Example: Let v1 = (0, 0, 1, 2) and v2 = (0, 1, 1). Then v1 ⋂ v2 = (0, 1), v1 ∪ v2 = (0, 0, 1, 1, 2). Then jaccard
    similarity equals |v1 ⋂ v2| / |v1 ∪ v2| = 0.4
    :param query_feat: first feature vector
    :param candidate_feat: second feature vector
    :return: similarity
    """
    intersect_size = _intersect(query_feat, candidate_feat)
    return float(intersect_size) / float(
        len(query_feat) + len(candidate_feat) - intersect_size
    )


@njit(nogil=True)
def _jaccard_overlap_threshold(n_elements: int, similarity_threshold: float) -> int:
    """
    It is a lower bound of `|a ⋂ b|` for any `b` which _jaccard(a, b) >= similarity_threshold
    :param n_elements: size of `a` features
    :param similarity_threshold: similarity threshold value
    :return: lower bound of `|a ⋂ b|`
    """
    return int(n_elements * similarity_threshold)


@njit(nogil=True)
def _jaccard_overlap_index_threshold(
    n_elements: int, similarity_threshold: float
) -> int:
    """
    It is a lower bound of `|a ⋂ b|` for any `b` which _jaccard(a, b) >= similarity_threshold
    This lower bound is used in creating prefix while indexing data
    :param n_elements: size of `a` features
    :param similarity_threshold: similarity threshold value
    :return: lower bound of `|a ⋂ b|`
    """
    return int(n_elements * similarity_threshold)


@njit(nogil=True)
def _jaccard_minoverlap(a_size: int, b_size: int, similarity_threshold: float) -> float:
    """
    It is an equivalent threshold for `|a ⋂ b|`, i.e. |a ⋂ b| >= minoverlap <=> _jaccard(a, b) >= similarity_threshold
    :param a_size: size of first feature vector
    :param b_size: size of second feature vector
    :param similarity_threshold: similarity threshold value
    :return: equivalent threshold for `|a ⋂ b|`
    """
    return float64(
        (similarity_threshold / (1.0 + similarity_threshold)) * (a_size + b_size)
    )


@njit(nogil=True)
def _cosine(query_feat: np.ndarray, candidate_feat: np.ndarray) -> float:
    """
    Cosine similarity for sorted vectors
    Example: Let v1 = (0, 0, 1, 2) and v2 = (0, 1, 1). Then v1 ⋂ v2 = (0, 1) and |v1| = 4, |v2| = 3. Then cosine
    similarity equals |v1 ⋂ v2| / sqrt(|v1| * |v2|) = 1/sqrt(3) ~ 0.58
    :param query_feat: first feature vector
    :param candidate_feat: second feature vector
    :return: similarity
    """
    intersect_size = _intersect(query_feat, candidate_feat)
    return float(intersect_size) / np.sqrt(float(len(query_feat) * len(candidate_feat)))


@njit(nogil=True)
def _cosine_overlap_threshold(n_elements: int, similarity_threshold: float) -> int:
    """
    It is a lower bound of `|a ⋂ b|` for any `b` which _cosine(a, b) >= similarity_threshold
    :param n_elements: size of `a` features
    :param similarity_threshold: similarity threshold value
    :return: lower bound of `|a ⋂ b|`
    """
    return int(np.sqrt(n_elements) * similarity_threshold)


@njit(nogil=True)
def _cosine_overlap_index_threshold(
    n_elements: int, similarity_threshold: float
) -> int:
    """
    It is a lower bound of `|a ⋂ b|` for any `b` which _cosine(a, b) >= similarity_threshold
    This lower bound is used in creating prefix while indexing data
    :param n_elements: size of `a` features
    :param similarity_threshold: similarity threshold value
    :return: lower bound of `|a ⋂ b|`
    """
    return int(np.sqrt(n_elements) * similarity_threshold)


@njit(nogil=True)
def _cosine_minoverlap(a_size: int, b_size: int, similarity_threshold: float) -> float:
    """
    It is an equivalent threshold for `|a ⋂ b|`, i.e. |a ⋂ b| >= minoverlap <=> _cosine(a, b) >= similarity_threshold
    :param a_size: size of first feature vector
    :param b_size: size of second feature vector
    :param similarity_threshold: similarity threshold value
    :return: equivalent threshold for `|a ⋂ b|`
    """
    return similarity_threshold * np.sqrt(a_size * b_size)


@njit(nogil=True)
def _containment_min(query_feat: np.ndarray, candidate_feat: np.ndarray) -> float:
    """
    Containment_min similarity for sorted vectors
    Example: Let v1 = (0, 0, 1, 2) and v2 = (0, 1, 1). Then v1 ⋂ v2 = (0, 1) and |v1| = 4, |v2| = 3. Then
    containment_min similarity equals |v1 ⋂ v2| / max(|v1|, |v2|) = 0.5
    :param query_feat: first feature vector
    :param candidate_feat: second feature vector
    :return: similarity
    """
    intersect_size = _intersect(query_feat, candidate_feat)
    return float(intersect_size) / float(max(len(query_feat), len(candidate_feat)))


@njit(nogil=True)
def _containment_min_overlap_threshold(
    n_elements: int, similarity_threshold: float
) -> int:
    """
    It is a lower bound of `|a ⋂ b|` for any `b` which _containment_min(a, b) >= similarity_threshold
    :param n_elements: size of `a` features
    :param similarity_threshold: similarity threshold value
    :return: lower bound of `|a ⋂ b|`
    """
    return int(n_elements * similarity_threshold)


@njit(nogil=True)
def _containment_min_overlap_index_threshold(
    n_elements: int, similarity_threshold: float
) -> int:
    """
    It is a lower bound of `|a ⋂ b|` for any `b` which _containment_min(a, b) >= similarity_threshold
    This lower bound is used in creating prefix while indexing data
    :param n_elements: size of `a` features
    :param similarity_threshold: similarity threshold value
    :return: lower bound of `|a ⋂ b|`
    """
    return int(n_elements * similarity_threshold)


@njit(nogil=True)
def _containment_min_minoverlap(
    a_size: int, b_size: int, similarity_threshold: float
) -> float:
    """
    It is an equivalent threshold for `|a ⋂ b|`, i.e. |a ⋂ b| >= minoverlap <=>
     _containment_min(a, b) >= similarity_threshold
    :param a_size: size of first feature vector
    :param b_size: size of second feature vector
    :param similarity_threshold: similarity threshold value
    :return: equivalent threshold for `|a ⋂ b|`
    """
    return similarity_threshold * max(a_size, b_size)


@njit(nogil=True)
def _containment(query_feat: np.ndarray, candidate_feat: np.ndarray) -> float:
    """
    Containment similarity for sorted vectors
    Example: Let v1 = (0, 0, 1, 2) and v2 = (0, 1, 1). Then v1 ⋂ v2 = (0, 1) and |v1| = 4, |v2| = 3. Then containment
    similarity equals |v1 ⋂ v2| / |v1| = 0.5
    :param query_feat: first feature vector
    :param candidate_feat: second feature vector
    :return: similarity
    """
    intersect_size = _intersect(query_feat, candidate_feat)
    return float(intersect_size) / float(len(query_feat))


@njit(nogil=True)
def _containment_overlap_threshold(n_elements: int, similarity_threshold: float) -> int:
    """
    It is a lower bound of `|a ⋂ b|` for any `b` which _containment(a, b) >= similarity_threshold
    :param n_elements: size of `a` features
    :param similarity_threshold: similarity threshold value
    :return: lower bound of `|a ⋂ b|`
    """
    return int(n_elements * similarity_threshold)


@njit(nogil=True)
def _containment_overlap_index_threshold(
    n_elements: int, similarity_threshold: float
) -> int:  # noqa: U100
    """
    It is a special lower bound of `|a ⋂ b|` for any `b` which _containment(a, b) >= similarity_threshold
    This lower bound is used in creating prefix while indexing. Due to containment is asymmetric function we have to
    index all the collection, so prefix must equal all the collection
    :param n_elements: size of `a` features
    :param similarity_threshold: similarity threshold value
    :return: lower bound of `|a ⋂ b|`
    """
    return 1


@njit(nogil=True)
def _containment_minoverlap(
    a_size: int, b_size: int, similarity_threshold: float
) -> float:  # noqa: U100
    """
    It is an equivalent threshold for `|a ⋂ b|`, i.e. |a ⋂ b| >= minoverlap <=>
     _containment(a, b) >= similarity_threshold
    :param a_size: size of first feature vector
    :param b_size: size of second feature vector
    :param similarity_threshold: similarity threshold value
    :return: equivalent threshold for `|a ⋂ b|`
    """
    return a_size * similarity_threshold


def _create_get_prefix_size_query(overlap_threshold_func: Callable) -> Callable:
    @njit(nogil=True)
    def _get_prefix_size_query(n_elements: int, similarity_threshold: float) -> int:
        """
        Evaluate the size of prefix for features in order to use it in querying
        :param n_elements: size of features
        :param similarity_threshold: similarity threshold value
        :return:
        """
        overlap_threshold = overlap_threshold_func(
            n_elements=n_elements, similarity_threshold=similarity_threshold
        )
        return n_elements - overlap_threshold + 1

    return _get_prefix_size_query


_jaccard_get_prefix_size_query = _create_get_prefix_size_query(
    _jaccard_overlap_threshold
)
_cosine_get_prefix_size_query = _create_get_prefix_size_query(_cosine_overlap_threshold)
_containment_min_get_prefix_size_query = _create_get_prefix_size_query(
    _containment_min_overlap_threshold
)
_containment_get_prefix_size_query = _create_get_prefix_size_query(
    _containment_overlap_threshold
)


def _create_position_filter(minoverlap_func: Callable) -> Callable:
    @njit(nogil=True)
    def _position_filter(
        a_size: int,
        b_size: int,
        a_prefix_pos: int,
        b_prefix_pos: int,
        similarity_threshold: float,
    ) -> bool:
        """
        Check position filter condition due to "Position-Enhanced Length Filter for Set Similarity Joins"
        :param a_size: size of first feature array
        :param b_size: size of second feature array
        :param a_prefix_pos: prefix position for first array
        :param b_prefix_pos: prefix position for second array
        :param similarity_threshold: similarity threshold
        :return: True if position filter condition satisfied else False
        """
        minoverlap_value = minoverlap_func(a_size, b_size, similarity_threshold)
        min_value = float64(min(a_size - a_prefix_pos, b_size - b_prefix_pos))
        greater_condition = float64(min_value) > float64(minoverlap_value)
        equal_condition = is_close(
            min_value, minoverlap_value, rel_tol=1e-09, abs_tol=0.0
        )
        return greater_condition or equal_condition

    return _position_filter


_jaccard_position_filter = _create_position_filter(_jaccard_minoverlap)
_cosine_position_filter = _create_position_filter(_cosine_minoverlap)
_containment_min_position_filter = _create_position_filter(_containment_min_minoverlap)
_containment_position_filter = _create_position_filter(_containment_minoverlap)


def _create_verify_pair(similarity_function: Callable) -> Callable:
    @njit(nogil=True)
    def _verify_pair(
        query_feat: np.ndarray, candidate_feat: np.ndarray, similarity_threshold: float
    ) -> bool:
        """
        Check if pair similarity not less than similarity threshold
        :param similarity_threshold: similarity threshold to check
        :param query_feat: input vector which compare to all other candidates
        :param candidate_feat: feature vector of candidate
        :return: `(index, similarity_score)` for truly similar vectors
        """
        sim_score = similarity_function(query_feat, candidate_feat)
        return (sim_score > similarity_threshold) or is_close(
            sim_score, similarity_threshold, rel_tol=1e-09, abs_tol=0.0
        )

    return _verify_pair


_jaccard_verify_pair = _create_verify_pair(_jaccard)
_cosine_verify_pair = _create_verify_pair(_cosine)
_containment_min_verify_pair = _create_verify_pair(_containment_min)
_containment_verify_pair = _create_verify_pair(_containment)


def _create_calculate_index_size(overlap_index_threshold_func: Callable) -> Callable:
    @njit(nogil=True, fastmath=True, parallel=True)
    def calculate_index_size(
        pointers: np.ndarray, prefix_sizes: np.ndarray, similarity_threshold: float
    ) -> int:
        """
        Evaluate prefix sizes for all vectors and size of index
        :param pointers: pointers of start and end for each features in data
        :param prefix_sizes: array which will be filled with prefix sizes
        :param similarity_threshold: similarity threshold value
        :return: size of index
        """
        total_prefix_size = 0
        for idx in prange(pointers.shape[0]):
            start = pointers[idx, 0]
            end = pointers[idx, 1]
            set_len = end - start
            overlap_threshold = overlap_index_threshold_func(
                n_elements=set_len, similarity_threshold=similarity_threshold
            )
            prefix_size = set_len - overlap_threshold + 1
            prefix_sizes[idx] = prefix_size
            total_prefix_size += prefix_size
        return total_prefix_size

    return calculate_index_size


jaccard_calculate_index_size = _create_calculate_index_size(
    _jaccard_overlap_index_threshold
)
cosine_calculate_index_size = _create_calculate_index_size(
    _cosine_overlap_index_threshold
)
containment_min_calculate_index_size = _create_calculate_index_size(
    _containment_min_overlap_index_threshold
)
containment_calculate_index_size = _create_calculate_index_size(
    _containment_overlap_index_threshold
)


# TODO: make documentation for SSS with pictures
@njit(nogil=True)
def _calc_feature_index_size(
    feature_index_size: np.ndarray,
    current_feature_pos: np.ndarray,
    input_pointer: np.ndarray,
    input_data: np.ndarray,
    prefix_sizes: np.ndarray,
) -> None:
    """
    Calculate start and end positions for each feature in index
    :param feature_index_size: empty matrix (n_features, 2) which will be filled
    :param current_feature_pos: empty array (n_features,) which will maintain position for each feature in index
    :param input_pointer: matrix with start and end positions of features in input_data for each vector
    :param input_data: features of all vectors
    :param prefix_sizes: size of prefix for each feature vector
    """
    for idx, (start, end) in enumerate(input_pointer):
        features = input_data[start:end]
        prefix_size = prefix_sizes[idx]
        prefix = features[:prefix_size]
        for feature_id in prefix:
            current_feature_pos[feature_id] += 1

    feature_index_size_counter = 0
    for idx in range(feature_index_size.shape[0]):
        feature_index_size[idx, 0] = feature_index_size_counter
        feature_index_size_counter += current_feature_pos[idx]
        feature_index_size[idx, 1] = feature_index_size_counter

    for idx in range(current_feature_pos.shape[0]):
        current_feature_pos[idx] = feature_index_size[idx, 0]


@njit(nogil=True)
def fill_index(
    index: np.ndarray,
    feature_index_size: np.ndarray,
    current_feature_pos: np.ndarray,
    input_pointer: np.ndarray,
    input_data: np.ndarray,
    prefix_sizes: np.ndarray,
) -> None:
    """
    Fill data structures for index given input data
    :param current_feature_pos: array which maintains current position for each feature in index
    :param prefix_sizes: size of prefix for each feature vector
    :param index: empty matrix (index_size, 2) which will be filled
    :param feature_index_size: matrix with start and end positions for each feature in index
    :param input_pointer: matrix with start and end positions of features in input_data for each vector
    :param input_data: features of all vectors
    """
    _calc_feature_index_size(
        feature_index_size=feature_index_size,
        current_feature_pos=current_feature_pos,
        input_pointer=input_pointer,
        input_data=input_data,
        prefix_sizes=prefix_sizes,
    )

    for feature_index, (start, end) in enumerate(input_pointer):
        features = input_data[start:end]
        prefix_size = prefix_sizes[feature_index]
        prefix = features[:prefix_size]
        for feature_prefix_pos, feature_id in enumerate(prefix):
            current_feature_idx = current_feature_pos[feature_id]
            index[current_feature_idx, 0] = feature_index
            index[current_feature_idx, 1] = feature_prefix_pos
            current_feature_pos[feature_id] += 1


@njit(nogil=True)
def _n_unique(data: np.ndarray) -> int:
    """
    Calculate number of unique encoded features
    :param data: integer features of all vectors
    :return: number of unique encoded features
    """
    counter = np.zeros_like(data)
    for feature in data:
        counter[feature] = 1
    n_features = 0
    for i in range(len(counter)):
        n_features += counter[i]
    return n_features


@njit(nogil=True)
def _fill_counter(counter: np.ndarray, data: np.ndarray) -> None:
    """
    Calculate amount for each integer feature
    :param counter: empty array (n_features,) which will be filled
    :param data: integer features of all vectors
    """
    for feature in data:
        counter[feature] += 1


@njit(nogil=True)
def _create_frequency_order_mapping(counter: np.ndarray) -> np.ndarray:
    """
    Calculate feature_to_frequency_id mapping due to frequency order
    :param counter: array with count for each encoded feature
    :return: array mapping encoded feature to its frequency id
    """
    frequency_sorted_features = np.argsort(counter)
    feature_to_frequency_id = np.zeros_like(frequency_sorted_features)
    for idx in range(len(frequency_sorted_features)):
        feature_to_frequency_id[frequency_sorted_features[idx]] = idx
    return feature_to_frequency_id


@njit(nogil=True)
def _apply_frequency_order_transform(
    data: np.ndarray,
    pointers: np.ndarray,
    feature_to_frequency_id: np.ndarray,
    fot_data: np.ndarray,
) -> None:
    """
    Apply frequency order transformation to integer data using feature_to_frequency_id - frequency order mapping
    :param data: integer features of vectors
    :param pointers: matrix with start and end positions of integer features in data for each vector
    :param feature_to_frequency_id: mapping encoded features from data to its ids due to frequency order
    :param fot_data: array shaped like data where frequency order transformed data will be written to
    """
    unknown_feature_id = len(
        feature_to_frequency_id
    )  # in case of new features not presented in index
    for pointer in pointers:
        start, end = pointer
        data_to_transform = data[start:end]
        for idx in range(len(data_to_transform)):
            feature = data_to_transform[idx]
            if feature >= len(feature_to_frequency_id):
                data_to_transform[idx] = unknown_feature_id
            else:
                data_to_transform[idx] = feature_to_frequency_id[feature]
        fot_data[start:end] = np.sort(data_to_transform)


def frequency_order_transform_from_flattened_format(
    data: np.ndarray, pointers: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply frequency order transformation for flattened format data
    :param data: flattened integer data in one array
    :param pointers: start and end positions inside data for each vector
    :return: frequency order transformed data, start and end positions for data, mapping encoded features to ids due to
    frequency order
    """
    n_features = _n_unique(data=data)
    feature2freq = np.zeros((n_features,), dtype=int)
    _fill_counter(counter=feature2freq, data=data)
    feature_to_frequency_id = _create_frequency_order_mapping(feature2freq)
    _apply_frequency_order_transform(
        data=data,
        pointers=pointers,
        feature_to_frequency_id=feature_to_frequency_id,
        fot_data=data,
    )
    return data, pointers, feature_to_frequency_id


def frequency_order_transform(
    index_dataset: Iterable[np.ndarray],
    features_lengths: np.ndarray,
    mmap: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply frequency order transformation for numeric transformed dataset
    :param index_dataset: iterable integer features of vectors
    :param features_lengths: array of lengths for each feature vector from index_dataset
    :param mmap: if mmap index_data or not
    :return: frequency order transformed data, start and end positions for data, mapping encoded features to ids due to
    frequency order
    """
    data, pointers = _to_flattened_format(
        dataset=index_dataset,
        features_lengths=features_lengths,
        mmap=mmap,
        mmap_filename="index_data_mmap",
    )
    return frequency_order_transform_from_flattened_format(data, pointers)


def frequency_order_transform_queries(
    queries_dataset: Sequence[np.ndarray], feature_to_frequency_id: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply frequency order transformation for numeric transformed query dataset using feature_to_frequency_id - mapping
    calculated while frequency order transforming dataset for indexing
    :param queries_dataset: sequence of integer features of query vectors
    :param feature_to_frequency_id: mapping encoded features to ids for already indexed data
    :return: frequency order transformed query data, start and end positions for query data
    """
    features_lengths = np.array(
        [len(features) for features in queries_dataset], dtype=np.int32
    )
    queries_data, pointers = _to_flattened_format(
        queries_dataset, features_lengths=features_lengths
    )
    _apply_frequency_order_transform(
        feature_to_frequency_id=feature_to_frequency_id,
        data=queries_data,
        pointers=pointers,
        fot_data=queries_data,
    )
    return queries_data, pointers


def _to_flattened_format(
    dataset: Iterable[np.ndarray],
    features_lengths: np.ndarray,
    mmap: bool = False,
    mmap_filename: str = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform data to flattened format (made for `numba`) - one array and pointers for start and end positions in data
    [[1, 2, 3], [2, 4, 1, 2], [1, 2]] -> ([1, 2, 3, 2, 4, 1, 2, 1, 2], [[0, 3], [3, 7], [7, 9]])
    :param dataset: iterable vectors features
    :param features_lengths: array of lengths for each feature vector from dataset
    :param mmap: if mmap flattened data or not
    :param mmap_filename: filename for mmap data
    :return: vector features flat mapped in one array, start and end positions for each feature vector
    """
    process = psutil.Process(os.getpid())
    logging.debug(
        f"To_flattened_format Starting: memory usage is {process.memory_info().rss / 1024**2}"
    )
    if mmap and mmap_filename is None:
        raise ValueError(
            "If mmap flag is True then mmap_filename argument should be provided"
        )
    if mmap:
        data = np.memmap(
            filename=mmap_filename,
            dtype=np.int32,
            mode="w+",
            shape=(np.sum(features_lengths),),
        )
    else:
        data = np.zeros(shape=(np.sum(features_lengths),), dtype=np.int32)
    pointers = np.zeros(shape=(len(features_lengths), 2), dtype=np.int32)
    current_features_index = 0
    offset = 0
    for features in dataset:
        data[offset : offset + features_lengths[current_features_index]] = features
        pointers[current_features_index, 0] = offset
        offset += features_lengths[current_features_index]
        pointers[current_features_index, 1] = offset
        current_features_index += 1
    logging.debug(
        f"To_flattened_format Finished: memory usage is {process.memory_info().rss / 1024 ** 2}"
    )
    return data, pointers


SIMILARITY_METRICS = {
    "jaccard": _jaccard,
    "cosine": _cosine,
    "containment_min": _containment_min,
    "containment": _containment,
}

SYMMETRIC_SIMILARITY_METRICS = ["jaccard", "cosine", "containment_min"]
ASYMMETRIC_SIMILARITY_METRICS = ["containment"]

"""
Provide a lower bound of `|a ⋂ b|` for any `b` which _similarity_function(a, b) >= similarity_threshold
"""
_LOWER_BOUND_OVERLAP_FUNCS = {
    "jaccard": _jaccard_overlap_threshold,
    "cosine": _cosine_overlap_threshold,
    "containment_min": _containment_min_overlap_threshold,
    "containment": _containment_overlap_threshold,
}

"""
Evaluate the size of prefix for features in order to use it in querying
"""
_GET_PREFIX_SIZE_QUERY_FUNCS = {
    "jaccard": _jaccard_get_prefix_size_query,
    "cosine": _cosine_get_prefix_size_query,
    "containment_min": _containment_min_get_prefix_size_query,
    "containment": _containment_get_prefix_size_query,
}

"""
Check position filter condition due to "Position-Enhanced Length Filter for Set Similarity Joins"
"""
_POSITION_FILTER_FUNCS = {
    "jaccard": _jaccard_position_filter,
    "cosine": _cosine_position_filter,
    "containment_min": _containment_min_position_filter,
    "containment": _containment_position_filter,
}

"""
Check if pair similarity not less than similarity threshold
"""
_VERIFY_SIMILARITY_FUNCS = {
    "jaccard": _jaccard_verify_pair,
    "cosine": _cosine_verify_pair,
    "containment_min": _containment_min_verify_pair,
    "containment": _containment_verify_pair,
}

"""
Evaluate prefix sizes for all vectors and size of whole index
"""
_CALCULATE_INDEX_SIZE_FUNCS = {
    "jaccard": jaccard_calculate_index_size,
    "cosine": cosine_calculate_index_size,
    "containment_min": containment_min_calculate_index_size,
    "containment": containment_calculate_index_size,
}


def query(
    similarity_metric_name: str,
    query_features: np.ndarray,
    index: np.ndarray,
    feature_index_size: np.ndarray,
    index_data: np.ndarray,
    index_pointers: np.ndarray,
    similarity_threshold: float,
) -> Set[Tuple[int, float]]:
    """
    Query features to indexed data
    :param similarity_metric_name: name of similarity metric
    :param query_features: frequency order transformed feature vector which will be in query
    :param index: index built on data
    :param feature_index_size: matrix with start and end positions for each feature in index
    :param index_data: data which is in index
    :param index_pointers: pointers of start and end for each features in indexed_data
    :param similarity_threshold: similarity threshold value for query
    :return: set of tuples `(candidate_index, similarity_score)` for truly similar vectors
    """
    index = np.asarray(index)
    feature_index_size = np.asarray(feature_index_size)
    index_data = np.asarray(index_data)
    index_pointers = np.asarray(index_pointers)

    get_prefix_size_query_func = _GET_PREFIX_SIZE_QUERY_FUNCS[similarity_metric_name]
    position_filter_func = _POSITION_FILTER_FUNCS[similarity_metric_name]
    verify_pair_func = _VERIFY_SIMILARITY_FUNCS[similarity_metric_name]
    similarity_func = SIMILARITY_METRICS[similarity_metric_name]

    result = set()
    prefix_size = get_prefix_size_query_func(
        n_elements=len(query_features), similarity_threshold=similarity_threshold
    )
    prefix = query_features[:prefix_size]

    query_prefix_size = len(query_features)
    candidates_to_check = 0
    for feature_id_pos in range(len(prefix)):
        feature_id = prefix[feature_id_pos]
        if feature_id >= feature_index_size.shape[0]:
            continue
        index_start = feature_index_size[feature_id, 0]
        index_end = feature_index_size[feature_id, 1]
        index_to_check = index[index_start:index_end]
        for idx in range(len(index_to_check)):
            candidate_idx = index_to_check[idx, 0]
            candidate_feature_id_pos = index_to_check[idx, 1]
            candidate_start = index_pointers[candidate_idx, 0]
            candidate_end = index_pointers[candidate_idx, 1]
            candidate_features = index_data[candidate_start:candidate_end]
            candidate_len = candidate_end - candidate_start
            candidates_to_check += 1
            if position_filter_func(
                a_size=query_prefix_size,
                b_size=candidate_len,
                a_prefix_pos=feature_id_pos,
                b_prefix_pos=candidate_feature_id_pos,
                similarity_threshold=similarity_threshold,
            ) and verify_pair_func(
                query_feat=query_features,
                candidate_feat=candidate_features,
                similarity_threshold=similarity_threshold,
            ):
                similarity_score = similarity_func(query_features, candidate_features)
                result.add((candidate_idx, similarity_score))

    logging.debug(
        f"Candidates to check {candidates_to_check}\nResults len for this one query {len(result)}"
    )
    return result


def query_batch(
    similarity_func_name: str,
    query_data: np.ndarray,
    query_pointers: np.ndarray,
    index: np.ndarray,
    feature_index_size: np.ndarray,
    index_data: np.ndarray,
    index_pointers: np.ndarray,
    similarity_threshold: float,
    show_progress: bool,
    n_cores: int,
) -> Set[Tuple[int, int, float]]:
    """
    Query bath of features to indexed data
    :param similarity_func_name: name of similarity function
    :param query_data: flattened features of query vectors
    :param query_pointers: pointers of start and end for each features in query_data
    :param index: index built on data
    :param feature_index_size: matrix with start and end positions for each feature in index
    :param index_data: data which is in index
    :param index_pointers: pointers of start and end for each features in indexed_data
    :param similarity_threshold: similarity threshold value for querying
    :param show_progress: flag to use `tqdm` progress bar or not
    :param n_cores: number of cores to use while processing many queries to index
    :return: set of tuples `(index_from_querying, index_from_data, similarity_score)` for truly similar vectors
    """

    def job_task(
        similarity_func_name,
        query_data_shared,
        query_pointers_shared,
        query_range,
        index_shared,
        feature_index_size_shared,
        index_data_shared,
        index_pointers_shared,
        similarity_threshold,
        queue,
    ):
        results = []
        for idx in tqdm(
            query_range, "Querying", disable=not show_progress, leave=False
        ):
            query_res = query(
                similarity_metric_name=similarity_func_name,
                query_features=query_data_shared[
                    query_pointers_shared[idx, 0] : query_pointers_shared[idx, 1]
                ],
                index=index_shared,
                feature_index_size=feature_index_size_shared,
                index_data=index_data_shared,
                index_pointers=index_pointers_shared,
                similarity_threshold=similarity_threshold,
            )
            for candidate_idx, similarity_score in query_res:
                results.append((idx, candidate_idx, similarity_score))
        for res in results:
            queue.put(res)
        # "end" of sending results
        queue.put(-1)

    def split_work(process_num: int):
        # round-robin shuffle
        query_ranges = []
        for idx in range(process_num):
            query_ranges.append(list(range(idx, len(query_pointers), process_num)))

        # share data structures for multiprocessing
        query_data_shared = (
            query_data
            if type(query_data) is np.memmap
            else sharedmem.full_like(query_data, query_data)
        )
        query_pointers_shared = (
            query_pointers
            if type(query_pointers) is np.memmap
            else sharedmem.full_like(query_pointers, query_pointers)
        )
        index_shared = (
            index if type(index) is np.memmap else sharedmem.full_like(index, index)
        )
        index_data_shared = (
            index_data
            if type(index_data) is np.memmap
            else sharedmem.full_like(index_data, index_data)
        )
        feature_index_size_shared = (
            feature_index_size
            if type(feature_index_size) is np.memmap
            else sharedmem.full_like(feature_index_size, feature_index_size)
        )
        index_pointers_shared = (
            index_pointers
            if type(index_pointers) is np.memmap
            else sharedmem.full_like(index_pointers, index_pointers)
        )

        queues = [Queue() for _ in range(len(query_ranges))]
        processes = [
            Process(
                target=job_task,
                args=(
                    similarity_func_name,
                    query_data_shared,
                    query_pointers_shared,
                    query_ranges[idx],
                    index_shared,
                    feature_index_size_shared,
                    index_data_shared,
                    index_pointers_shared,
                    similarity_threshold,
                    queues[idx],
                ),
            )
            for idx in range(len(query_ranges))
        ]
        for p in processes:
            p.start()

        # Getting results from queues in cyclic way
        result_set = set()
        finished_queues_ids = np.zeros((process_num,), dtype=bool)
        finished_queues_num = 0
        queue_id = 0
        while finished_queues_num < process_num:
            if not finished_queues_ids[queue_id]:
                get_val = queues[queue_id].get()
                if get_val == -1:
                    finished_queues_ids[queue_id] = True
                    finished_queues_num += 1
                else:
                    result_set.add(get_val)

            # switch to next not finished queue id
            if finished_queues_num >= process_num:
                break
            while finished_queues_ids[queue_id]:
                queue_id = (queue_id + 1) % process_num

        for p in processes:
            p.join()

        return result_set

    result = split_work(n_cores)
    return result


# Init JIT Functions
is_close(a=0.001, b=0.002, rel_tol=1e-09, abs_tol=0.0)
_intersect(
    a=np.array([0, 1, 1, 2], dtype=np.int32), b=np.array([1, 1, 2], dtype=np.int32)
)

_jaccard(
    query_feat=np.array([0, 1, 1, 2], dtype=np.int32),
    candidate_feat=np.array([1, 1, 2], dtype=np.int32),
)
_cosine(
    query_feat=np.array([0, 1, 1, 2], dtype=np.int32),
    candidate_feat=np.array([1, 1, 2], dtype=np.int32),
)
_containment_min(
    query_feat=np.array([0, 1, 1, 2], dtype=np.int32),
    candidate_feat=np.array([1, 1, 2], dtype=np.int32),
)
_containment(
    query_feat=np.array([0, 1, 1, 2], dtype=np.int32),
    candidate_feat=np.array([1, 1, 2], dtype=np.int32),
)

_jaccard_overlap_threshold(n_elements=12, similarity_threshold=0.8)
_cosine_overlap_threshold(n_elements=12, similarity_threshold=0.8)
_containment_min_overlap_threshold(n_elements=12, similarity_threshold=0.8)
_containment_overlap_threshold(n_elements=12, similarity_threshold=0.8)

_jaccard_overlap_index_threshold(n_elements=12, similarity_threshold=0.8)
_cosine_overlap_index_threshold(n_elements=12, similarity_threshold=0.8)
_containment_min_overlap_index_threshold(n_elements=12, similarity_threshold=0.8)
_containment_overlap_index_threshold(n_elements=12, similarity_threshold=0.8)

_jaccard_minoverlap(a_size=12, b_size=14, similarity_threshold=0.8)
_cosine_minoverlap(a_size=12, b_size=14, similarity_threshold=0.8)
_containment_min_minoverlap(a_size=12, b_size=14, similarity_threshold=0.8)
_containment_minoverlap(a_size=12, b_size=14, similarity_threshold=0.8)

_jaccard_get_prefix_size_query(n_elements=12, similarity_threshold=0.8)
_cosine_get_prefix_size_query(n_elements=12, similarity_threshold=0.8)
_containment_min_get_prefix_size_query(n_elements=12, similarity_threshold=0.8)
_containment_get_prefix_size_query(n_elements=12, similarity_threshold=0.8)

_jaccard_position_filter(
    a_size=12, b_size=14, a_prefix_pos=2, b_prefix_pos=3, similarity_threshold=0.8
)
_cosine_position_filter(
    a_size=12, b_size=14, a_prefix_pos=2, b_prefix_pos=3, similarity_threshold=0.8
)
_containment_min_position_filter(
    a_size=12, b_size=14, a_prefix_pos=2, b_prefix_pos=3, similarity_threshold=0.8
)
_containment_position_filter(
    a_size=12, b_size=14, a_prefix_pos=2, b_prefix_pos=3, similarity_threshold=0.8
)

_jaccard_verify_pair(
    query_feat=np.array([0, 1, 1, 2], dtype=np.int32),
    candidate_feat=np.array([1, 1, 2], dtype=np.int32),
    similarity_threshold=0.8,
)
_cosine_verify_pair(
    query_feat=np.array([0, 1, 1, 2], dtype=np.int32),
    candidate_feat=np.array([1, 1, 2], dtype=np.int32),
    similarity_threshold=0.8,
)
_containment_min_verify_pair(
    query_feat=np.array([0, 1, 1, 2], dtype=np.int32),
    candidate_feat=np.array([1, 1, 2], dtype=np.int32),
    similarity_threshold=0.8,
)
_containment_verify_pair(
    query_feat=np.array([0, 1, 1, 2], dtype=np.int32),
    candidate_feat=np.array([1, 1, 2], dtype=np.int32),
    similarity_threshold=0.8,
)

jaccard_calculate_index_size(
    pointers=np.array([[0, 2], [2, 5], [5, 9]], dtype=np.int32),
    prefix_sizes=np.array([0, 0, 0], dtype=np.int32),
    similarity_threshold=0.8,
)
cosine_calculate_index_size(
    pointers=np.array([[0, 2], [2, 5], [5, 9]], dtype=np.int32),
    prefix_sizes=np.array([0, 0, 0], dtype=np.int32),
    similarity_threshold=0.8,
)
containment_min_calculate_index_size(
    pointers=np.array([[0, 2], [2, 5], [5, 9]], dtype=np.int32),
    prefix_sizes=np.array([0, 0, 0], dtype=np.int32),
    similarity_threshold=0.8,
)
containment_calculate_index_size(
    pointers=np.array([[0, 2], [2, 5], [5, 9]], dtype=np.int32),
    prefix_sizes=np.array([0, 0, 0], dtype=np.int32),
    similarity_threshold=0.8,
)

_calc_feature_index_size(
    feature_index_size=np.zeros((5, 2), dtype=np.int32),
    current_feature_pos=np.zeros((5,), dtype=np.int32),
    input_pointer=np.array([[0, 3], [3, 7]], dtype=np.int32),
    input_data=np.array([0, 3, 1, 2, 3, 1, 2, 4], dtype=np.int32),
    prefix_sizes=np.array([2, 3], dtype=np.int32),
)

fill_index(
    index=np.zeros((8, 2), dtype=np.int32),
    feature_index_size=np.zeros((8, 2), dtype=np.int32),
    current_feature_pos=np.zeros((8,), dtype=np.int32),
    input_pointer=np.array([[0, 7], [7, 14], [14, 19]], dtype=np.int32),
    input_data=np.array(
        [0, 3, 4, 6, 6, 7, 7, 1, 3, 4, 5, 5, 7, 7, 2, 6, 6, 6, 7], dtype=np.int32
    ),
    prefix_sizes=np.array([3, 3, 2], dtype=np.int32),
)

_n_unique(data=np.array([1, 2, 2, 1], dtype=np.int32))

_fill_counter(
    counter=np.zeros((2,), dtype=np.int32), data=np.array([1, 2, 2, 1], dtype=np.int32)
)

_create_frequency_order_mapping(counter=np.array([2, 4], dtype=np.int32))

_apply_frequency_order_transform(
    data=np.array([0, 2, 2, 4, 5, 1, 2, 3], dtype=np.int32),
    pointers=np.array([[0, 2], [2, 5], [5, 8]], dtype=np.int32),
    feature_to_frequency_id=np.array([0, 1, 5, 2, 3, 4], dtype=np.int32),
    fot_data=np.zeros((8,), dtype=np.int32),
)
