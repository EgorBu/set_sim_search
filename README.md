
# SetSimSearcher

## Overview

SetSimSearcher is a Python library designed for efficient and simple multiset similarity search, supporting various similarity metrics including Jaccard, Cosine, Containment, and Containment Min. It leverages advanced filtering techniques, such as prefix and position filters, inspired by research for optimizing search processes in large datasets. This library is especially useful for applications in data mining, information retrieval, and similar fields where identifying similar item sets is crucial.

## Features

- Supports multiple similarity metrics: Jaccard, Cosine, Containment, Containment Min.
- Efficient querying of large datasets using advanced filtering techniques.
- Utilizes `numba.jit` for performance-critical function optimizations, significantly speeding up calculations.
- Configurable similarity threshold and progress bar visualization.
- Multicore processing capabilities for handling large-scale queries.
- Ability to save and load pre-built indexes for repeated use.

## Installation

To install SetSimSearcher, clone this repository and install the required dependencies.

```bash
git clone https://github.com/EgorBu/set_sim_search.git
cd set_sim_search
pip install -r requirements.txt
```

## Quick Start

Intialize `SetSimSearcher` - select similarity metric and threshold
```python
from set_sim_search import SetSimSearcher

# Initialize the searcher
searcher = SetSimSearcher(similarity_metric="jaccard", similarity_threshold=0.5)
```

Prepare index dataset - and 
```python
# Example dataset
dataset = [["a", "b", "c"], ["d", "e", "f"], ["a", "c", "e"]]
entities_names = ["sample1", "sample2", "sample3"]  # optional - if skipped index of the row will be returned in results
# Build index
searcher.build_index(dataset, entities_names=entities_names)
```

You can query one sample
```python
# Query
results = searcher.query(["a", "b"])
print(results)
```
and result will be:
```python
{('sample1', 0.6666666666666666)}
```

Or you can query many samples:
```python
# Query batch
queries = [["a", "b"], ["d", "e"], ["f", "q"]]
queries_names = ["query1", "query2", "query3"]
results = searcher.query_many(queries, query_entities_names=queries_names)
print(results)
```
and result will be:
```python
{('query2', 'sample2', 0.6666666666666666), ('query1', 'sample1', 0.6666666666666666)}
```

### All-to-all search
You may need to find near duplicates in your dataset, in this case `all_to_all` function may help you:
```python
from set_sim_search import SetSimSearcher

# Initialize the searcher
searcher = SetSimSearcher(similarity_metric="jaccard", similarity_threshold=0.5)

# Example dataset
dataset = [["a", "b", "c"], ["d", "e", "f"], ["a", "c", "e"]]
entities_names = ["sample1", "sample2", "sample3"]

# All-to-all - you don't need to prepare index - it will be done automatically
results = searcher.all_to_all(dataset, entities_names=entities_names)
print(results)
```
and result is:
```python
{('sample3', 'sample1', 0.5)}
```


## Performance Optimization with Numba

SetSimSearcher uses `numba.jit` to optimize several performance-critical functions. Numba is a Just-In-Time (JIT) compiler that translates a subset of Python and NumPy code into fast machine code. This optimization is crucial for achieving significant speed-ups in similarity search calculations, especially when processing large datasets.

To ensure optimal performance, make sure numba is correctly installed and your system meets its requirements. For more information on numba and its capabilities, visit the [Numba documentation](http://numba.pydata.org/).

## Contributing

Contributions to SetSimSearcher are welcome! If you have suggestions for improvements or bug fixes, please open an issue or pull request.

## License

SetSimSearcher is released under the [MIT License](LICENSE.md).
