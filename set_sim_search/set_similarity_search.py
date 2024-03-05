import logging
import os
from pathlib import Path
import sys
from typing import Any, Hashable, Iterable, List, Sequence, Set, Tuple, Union

import numpy as np
import psutil

from set_sim_search import utils


class SetSimSearcher:
    """
    SetSimSearcher supports multiset similarity search queries for similarity metrics: "jaccard", "cosine",
    "containment", "containment_min".
    The algorithm is based on a combination of the prefix filter and position filter techniques based on article:
    https://people.cs.rutgers.edu/~dd903/assets/papers/sigmod19.pdf
    """

    INDEX_MMAP_FILENAME = "index_mmap"

    def __init__(
        self,
        similarity_metric: str = "jaccard",
        similarity_threshold: float = 0.9,
        enable_progress_bar: bool = False,
        n_cores: int = os.cpu_count(),
    ):
        """
        :param similarity_metric: the name of the similarity function used this function currently supports
                                  "jaccard", "cosine", "containment", "containment_min"
        :param similarity_threshold: the threshold used, must be in (0, 1]
        :param enable_progress_bar: flag to use `tqdm` progress bar or not
        :param n_cores: number of cores to use while processing many queries to index
        """
        if similarity_metric not in utils.SIMILARITY_METRICS:
            raise ValueError(f"Similarity metric {similarity_metric} is not supported")
        if not isinstance(similarity_threshold, float):
            raise TypeError(
                f"Similarity threshold must be float value, got {type(similarity_threshold)}"
            )
        if not 0 < similarity_threshold <= 1.0:
            raise ValueError("Similarity threshold must be in the range (0, 1]")
        self.similarity_metric = similarity_metric
        self.similarity_threshold = similarity_threshold
        self.enable_progress_bar = enable_progress_bar
        self.n_cores = n_cores
        self._init_empty_index()

    def query(self, features: Sequence[Hashable]) -> Set[Tuple[Union[Any, int], float]]:
        """
        Query features to index
        :param features: feature vector
        :return: list of tuples `(candidate_entity_name/candidate_index, similarity_score)` for truly similar vectors
        """
        query_features = self._encode_queries([features])
        query_data, _ = utils.frequency_order_transform_queries(
            queries_dataset=query_features,
            feature_to_frequency_id=self.feature_to_frequency_id,
        )
        result_set = utils.query(
            similarity_metric_name=self.similarity_metric,
            query_features=query_data,
            index=self.index,
            feature_index_size=self.feature_index_size,
            index_data=self.index_data,
            index_pointers=self.index_pointers,
            similarity_threshold=self.similarity_threshold,
        )
        return {
            (self.entities_names[candidate_index], similarity_score)
            for candidate_index, similarity_score in result_set
        }

    def query_many(
        self,
        query_dataset: Sequence[Sequence[Hashable]],
        query_entities_names: Sequence[Any] = None,
    ) -> Set[Tuple[Union[Any, int], Union[Any, int], float]]:
        """
        Query batch of vector features to index
        :param query_dataset: sequence of vector features will be in query
        :param query_entities_names: sequence of entities names for each query vector
        :return: set of tuples `(query_entity_name/index_from_query_dataset,
                                 indexed_entity_name/index_from_indexed_dataset,
                                 similarity_score
                                )` for truly similar feature vectors
        """
        if query_entities_names is None:
            query_entities_names = list(range(len(query_dataset)))
        numeric_query_features_dataset = self._encode_queries(query_dataset)
        query_data, query_pointers = utils.frequency_order_transform_queries(
            numeric_query_features_dataset, self.feature_to_frequency_id
        )
        result = utils.query_batch(
            similarity_func_name=self.similarity_metric,
            query_data=query_data,
            query_pointers=query_pointers,
            index=self.index,
            feature_index_size=self.feature_index_size,
            index_data=self.index_data,
            index_pointers=self.index_pointers,
            similarity_threshold=self.similarity_threshold,
            show_progress=self.enable_progress_bar,
            n_cores=self.n_cores,
        )
        return {
            (
                query_entities_names[index_from_query_data],
                self.entities_names[index_from_index_data],
                similarity_score,
            )
            for index_from_query_data, index_from_index_data, similarity_score in result
        }

    def all_to_all(
        self,
        dataset: Sequence[Sequence[Hashable]],
        entities_names: Sequence[Any] = None,
    ) -> Set[Tuple[int, int, float]]:
        """
        Find all pairs of similar vectors in dataset with similarity higher that self.similarity_threshold
        (exclude entities equal to itself)
        :param dataset: sequence of feature vectors
        :param entities_names: sequence of entities names for each feature vecto
        :return: set of tuples `(index_from_dataset, index_from_dataset, similarity_score)` for truly similar feature
                 vectors
        """
        self.build_index(index_dataset=dataset, entities_names=entities_names)
        pairs = self.query_many(
            query_dataset=dataset, query_entities_names=entities_names
        )
        is_symmetric_similarity_function = (
            self.similarity_metric in utils.SYMMETRIC_SIMILARITY_METRICS
        )

        result_pairs = set()
        for p in pairs:
            if p[0] != p[1]:
                if is_symmetric_similarity_function:
                    result_pairs.add((*sorted([p[0], p[1]], reverse=True), p[2]))
                else:
                    result_pairs.add(p)
        return result_pairs

    def save_index(self, save_directory: str) -> None:
        """
        Save index to directory
        :param save_directory: path to folder where index will be saved
        """
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / "index_data.npy", self.index_data)
        np.save(save_dir / "index_pointers.npy", self.index_pointers)
        np.save(save_dir / "token2id.npy", self.feature_to_frequency_id)
        np.save(save_dir / "index.npy", self.index)
        np.save(save_dir / "token_index_size.npy", self.feature_index_size)
        np.save(save_dir / "entities_names.npy", self.entities_names)
        np.save(save_dir / "feature_to_index.npy", self.feature_to_index)

    def load_index(self, index_directory: str, mmap: bool = False) -> None:
        """
        Load index from storage directory at disk
        :param index_directory: path to folder where index is stored
        :param mmap: if mmap loaded index and indexed_data or not
        """
        self._load_index(index_directory=index_directory, mmap=mmap)

    def build_index(
            self,
            index_dataset: Iterable[Sequence[Hashable]],
            features_lengths: np.ndarray = None,
            entities_names: Sequence[Any] = None,
            mmap: bool = False,
    ) -> None:
        """
        Build index for dataset
        :param index_dataset: iterable of vector features
        :param features_lengths: array of lengths for each feature vector from index_dataset
        :param entities_names: sequence of entities names for each feature vector
        :param mmap: if mmap index or not
        """
        logging.debug("Starting: building index step")
        process = psutil.Process(os.getpid())
        logging.debug(
            f"Process taken memory beginning build index MB {process.memory_info().rss / 1024 ** 2}"
        )

        logging.debug("Starting: checking features_lengths")
        if features_lengths is None:
            if isinstance(index_dataset, Sequence):
                features_lengths = np.array(
                    [len(features) for features in index_dataset], dtype=np.int32
                )
            else:
                raise ValueError("features_lengths argument should be provided")
        logging.debug("Finished: checking features_lengths")

        logging.debug("Starting: checking entities_names")
        if entities_names is None:
            entities_names = np.arange(len(features_lengths))
        self.entities_names = entities_names
        logging.debug("Finished: checking entities_names")

        logging.debug("Starting: encoding step")
        self.feature_to_index = dict()
        encoded_dataset_generator = self._encode_features(index_dataset)
        logging.debug("Finished: encoding step")

        logging.debug("Starting: frequency order transformation step")
        self.index_data, self.index_pointers, self.feature_to_frequency_id = (
            utils.frequency_order_transform(
                index_dataset=encoded_dataset_generator,
                features_lengths=features_lengths,
                mmap=mmap,
            )
        )
        logging.debug("Finished: frequency order transformation step")
        logging.debug(
            f"Taken memory after freq_order_trans MB {process.memory_info().rss / 1024 ** 2}"
        )

        logging.debug("Starting: init index structures step")
        self.feature_index_size, current_feature_pos, index_prefix_sizes, index_size = (
            self._init_index_structures()
        )
        logging.debug("Finished: init index structures step")

        logging.debug("Starting: fill index step")
        if mmap:
            self.index = np.memmap(
                filename=self.INDEX_MMAP_FILENAME,
                dtype=np.int32,
                mode="w+",
                shape=(index_size, 2),
            )
        else:
            self.index = np.zeros((index_size, 2), dtype=np.int32)
        utils.fill_index(
            index=self.index,
            feature_index_size=self.feature_index_size,
            current_feature_pos=current_feature_pos,
            input_pointer=self.index_pointers,
            input_data=self.index_data,
            prefix_sizes=index_prefix_sizes,
        )
        logging.debug("Finished: fill index step")

        logging.debug(
            f"Index size MB {self.index.nbytes / 1024 ** 2}; \n"
            f"Token_index_size size MB {self.feature_index_size.nbytes / 1024 ** 2}; \n"
            f"Index_prefix_size MB {index_prefix_sizes.nbytes / 1024 ** 2}; \n"
            f"Current_token_pos MB {current_feature_pos.nbytes / 1024 ** 2}; \n"
            f"Token2int_value size MB {sys.getsizeof(self.feature_to_index) / 1024 ** 2}; \n"
            f"Finished building index size MB {process.memory_info().rss / 1024 ** 2}"
        )
        logging.debug("Finished: building index step")

    def _init_empty_index(self):
        """Initialize empty datastructures for index"""
        self.index_data = None
        self.index_pointers = None
        self.index = None
        self.feature_to_frequency_id = None
        self.feature_index_size = None
        self.feature_to_index = None
        self.entities_names = None

    def _encode_features(
        self, dataset: Iterable[Sequence[Hashable]]
    ) -> Iterable[np.ndarray]:
        """
        Transform a dataset to a numeric format - assign index to each unique value and replace input values with index
        :param dataset: iterable of feature vectors like ["a", "b", "c", "c"], ["d", "b"], ...
        :return: generator of integer_dataset - [0, 1, 2, 2], [3, 1], ...
        """
        for features in dataset:
            yield np.array(
                [
                    self.feature_to_index.setdefault(
                        feature, len(self.feature_to_index)
                    )
                    for feature in features
                ],
                dtype=np.int32,
            )

    def _load_index(self, index_directory: str, mmap: bool = False) -> None:
        """
        Load index from storage directory at disk
        :param index_directory: path to folder where index is stored
        :param mmap: if mmap loaded index and indexed_data or not
        """
        index_dir = Path(index_directory)
        self.index_data = np.load(
            index_dir / "index_data.npy", mmap_mode="r" if mmap else None
        )
        self.index_pointers = np.load(index_dir / "index_pointers.npy")
        self.index = np.load(index_dir / "index.npy", mmap_mode="r" if mmap else None)
        self.feature_to_frequency_id = np.load(index_dir / "token2id.npy")
        self.feature_index_size = np.load(index_dir / "token_index_size.npy")
        self.entities_names = np.load(index_dir / "entities_names.npy")
        self.feature_to_index = np.load(
            index_dir / "feature_to_index.npy", allow_pickle=True
        ).item()

    def _init_index_structures(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Init index structures for index building step
        :return: feature_index_size - empty matrix with start and end positions for each token in index
                 current_feature_pos - empty array which will maintain current position for each token in index
                 index_prefix_sizes - size of prefix for each feature vector
                 index_size - size of index
        """
        n_tokens = len(self.feature_to_frequency_id)
        feature_index_size = np.zeros((n_tokens, 2), dtype=np.int32)
        current_feature_pos = np.zeros((n_tokens,), dtype=np.int32)
        index_prefix_sizes = np.zeros((self.index_pointers.shape[0],), dtype=np.int32)
        index_size = utils._CALCULATE_INDEX_SIZE_FUNCS[self.similarity_metric](
            pointers=self.index_pointers,
            prefix_sizes=index_prefix_sizes,
            similarity_threshold=self.similarity_threshold,
        )
        return feature_index_size, current_feature_pos, index_prefix_sizes, index_size

    def _encode_queries(
        self, query_dataset: Sequence[Sequence[Hashable]]
    ) -> List[np.ndarray]:
        """
        Transform queries to a numeric format - assign index to each unique value and replace input values with index
        :param query_dataset: sequence of vector features will be in query
        :return: numeric queries dataset
        """
        encoded_queries = []
        unknown_token_int_value = len(self.feature_to_index)
        for query_features in query_dataset:
            numeric_query_features = []
            for token in query_features:
                if token in self.feature_to_index:
                    numeric_query_features.append(self.feature_to_index[token])
                else:
                    numeric_query_features.append(unknown_token_int_value)
                    unknown_token_int_value += 1
            encoded_queries.append(np.array(numeric_query_features))
        return encoded_queries


