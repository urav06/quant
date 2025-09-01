# Angus Dempster, Daniel F Schmidt, Geoffrey I Webb

# QUANT: A Minimalist Interval Method for Time Series Classification
# https://arxiv.org/abs/2308.00928
# (update to handle multivariate per https://github.com/angus924/aaltd2024)

import math

import torch
import torch.nn.functional as F

# == Generate Intervals ========================================================

def make_intervals(input_length: int, depth: int) -> torch.Tensor:
    """ Generate dyadic intervals with shifted variants. """

    max_depth: int                      = min(depth, int(math.log2(input_length)) + 1)
    all_intervals: list[torch.Tensor]   = []

    for level in range(max_depth):

        num_intervals: int          = 2 ** level
        boundaries: torch.Tensor    = torch.linspace(0, input_length, num_intervals + 1).long()
        intervals: torch.Tensor     = torch.stack((boundaries[:-1], boundaries[1:]), 1)

        all_intervals.append(intervals)

        # Add shifted intervals only if typical interval length > 1
        if num_intervals > 1 and intervals.diff().median() > 1:
            shift_distance: int             = int(math.ceil(input_length / num_intervals / 2))
            shifted_intervals: torch.Tensor = intervals[:-1] + shift_distance
            all_intervals.append(shifted_intervals)

    return torch.cat(all_intervals)

# == Quantile Function =========================================================

def f_quantile(interval_data: torch.Tensor, quantile_divisor: int = 4) -> torch.Tensor:
    """ Extract quantiles from interval data. """
    
    interval_length = interval_data.shape[-1]

    # Edge case: single-value intervals just return the value as-is
    if interval_length == 1:
        return interval_data.view(interval_data.shape[0], 1, interval_data.shape[1] * interval_data.shape[2])
    
    num_quantiles = 1 + (interval_length - 1) // quantile_divisor
    
    if num_quantiles == 1:
        # Special case: formula yields single quantile, use median (0.5 quantile)
        quantile_positions  = torch.tensor([0.5])
        quantiles           = interval_data.quantile(quantile_positions, dim=-1).permute(1, 2, 0)
        return quantiles.view(quantiles.shape[0], 1, quantiles.shape[1] * quantiles.shape[2])
    
    else:
        # Main case: extract multiple evenly-spaced quantiles [0, 1/(k-1), 2/(k-1), ..., 1]
        quantile_positions      = torch.linspace(0, 1, num_quantiles)
        quantiles               = interval_data.quantile(quantile_positions, dim=-1).permute(1, 2, 0)
        quantiles[..., 1::2]    = quantiles[..., 1::2] - interval_data.mean(-1, keepdim=True) # Apply mean subtraction to every 2nd quantile
        return quantiles.view(quantiles.shape[0], 1, quantiles.shape[1] * quantiles.shape[2])

# == Interval Model (per representation) =======================================

class IntervalModel():
    """ Interval-based feature extractor. """

    def __init__(self, input_length: int, depth: int = 6, quantile_divisor: int = 4) -> None:

        if quantile_divisor < 1:
            raise ValueError(f"quantile_divisor must be >= 1, got {quantile_divisor}")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")

        self.quantile_divisor   = quantile_divisor
        self.intervals          = make_intervals(input_length, depth)

    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        pass

    def transform(self, X: torch.Tensor) -> torch.Tensor:

        extracted_features: list[torch.Tensor] = []

        for start, end in self.intervals:
            interval_data: torch.Tensor     = X[..., start:end]
            interval_features: torch.Tensor = f_quantile(interval_data, self.quantile_divisor).squeeze(1)
            extracted_features.append(interval_features)

        return torch.cat(extracted_features, -1)

    def fit_transform(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:

        self.fit(X, Y)
        return self.transform(X)

# == Quant =====================================================================

class Quant():
    """ QUANT: A Minimalist Interval Method for Time Series Classification. """

    def __init__(self, depth: int = 6, div: int = 4) -> None:

        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if div < 1:
            raise ValueError(f"quantile_divisor must be >= 1, got {div}")

        self.depth: int                         = depth
        self.div: int                           = div
        self.models: dict[int, 'IntervalModel'] = {}
        self.fitted: bool                       = False
        self.representation_functions: tuple    = (
            lambda X: X,
            lambda X: F.avg_pool1d(F.pad(X.diff(), (2, 2), "replicate"), 5, 1),
            lambda X: X.diff(n=2),
            lambda X: torch.fft.rfft(X).abs(),
        )

    def transform(self, X: torch.Tensor) -> torch.Tensor:

        if not self.fitted:
            raise RuntimeError("not fitted")

        extracted_features: list[torch.Tensor] = []

        for index, function in enumerate(self.representation_functions):
            Z = function(X)
            extracted_features.append(self.models[index].transform(Z))
        
        return torch.cat(extracted_features, dim=-1)
    
    def fit_transform(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:

        extracted_features: list[torch.Tensor] = []

        for index, function in enumerate(self.representation_functions):
            Z = function(X)
            self.models[index] = IntervalModel(Z.shape[-1], self.depth, self.div)
            features = self.models[index].fit_transform(Z, Y)
            extracted_features.append(features)
        
        self.fitted = True
        return torch.cat(extracted_features, dim=-1)
