"""Aggregation strategys."""

import os
from typing import List, Optional, Tuple

from flwr.common.typing import NDArrays
from flwr.server.strategy.aggregate import aggregate

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class BaseStrategy: # takes the parameters from each client and aggregates them
    """Base strategy."""

    def __init__(self) -> None:
        """Init."""
        pass

    def aggregate_fit_fedavg(
        self,results) -> Optional[NDArrays]:
        """Fedavg aggregation function."""
        # print("Aggregation started")

        # res_formatted = [(result[0]["parameters"], 1) for result in results]
        # the "1" is for weighting by e.g. number of training examples
        parameters_aggregated = aggregate(results)
        
        # print("Aggregation done")
        return parameters_aggregated, {}
# q: what does this code do?
# a: it takes the parameters from each client and aggregates them