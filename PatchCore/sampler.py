import abc
from typing import Union

import numpy as np
import torch
import tqdm


class IdentitySampler:
    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        return features


class BaseSampler(abc.ABC):
    def __init__(self, percentage: float):
        if not 0 < percentage < 1:
            raise ValueError("Percentage value not in (0, 1).")
        self.percentage = percentage

    @abc.abstractmethod
    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        pass

    def _store_type(self, features: Union[torch.Tensor, np.ndarray]) -> None:
        self.features_is_numpy = isinstance(features, np.ndarray)
        if not self.features_is_numpy:
            self.features_device = features.device

    def _restore_type(self, features: torch.Tensor) -> Union[torch.Tensor, np.ndarray]:
        if self.features_is_numpy:
            return features.cpu().numpy()
        return features.to(self.features_device)


class GreedyCoresetSampler(BaseSampler):
    def __init__(
        self,
        percentage: float,
        device: torch.device,
        dimension_to_project_features_to=128,
    ):
        """Greedy Coreset sampling base class."""
        super().__init__(percentage)

        self.device = device
        self.dimension_to_project_features_to = dimension_to_project_features_to

    def _reduce_features(self, features):
        if features.shape[1] == self.dimension_to_project_features_to:
            return features
        mapper = torch.nn.Linear(
            features.shape[1], self.dimension_to_project_features_to, bias=False
        )
        _ = mapper.to(self.device)
        features = features.to(self.device)
        return mapper(features)

    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Subsamples features using Greedy Coreset.
        Args:
            features: [N x D]
        """
        if self.percentage == 1:
            return features
        self._store_type(features)
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        reduced_features = self._reduce_features(features) # feature의 차수를 줄인다.
        sample_indices = self._compute_greedy_coreset_indices(reduced_features) # 선택된 샘플의 인덱스 정보 저장
        features = features[sample_indices]
        return self._restore_type(features)

    @staticmethod
    def _compute_batchwise_differences(
        matrix_a: torch.Tensor, matrix_b: torch.Tensor
    ) -> torch.Tensor:
        """Computes batchwise Euclidean distances using PyTorch."""
        a_times_a = matrix_a.unsqueeze(1).bmm(matrix_a.unsqueeze(2)).reshape(-1, 1)
        b_times_b = matrix_b.unsqueeze(1).bmm(matrix_b.unsqueeze(2)).reshape(1, -1)
        a_times_b = matrix_a.mm(matrix_b.T)

        return (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None).sqrt()

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        """Runs iterative greedy coreset selection.
        Args:
            features: [NxD] input feature bank to sample.
        """
        
        # 각 패치끼리의 거리를 구한다.
        distance_matrix = self._compute_batchwise_differences(features, features)
        
        # 한 패치에 대하여, 다른 패치까지의 norm을 구해서 anchor로 삼는다.
        # 즉, anchor가 크다는 의미는 그 패치가 다른 패치들과 동떨어져 있다는 의미이다.
        coreset_anchor_distances = torch.norm(distance_matrix, dim=1)

        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percentage)

        for _ in range(num_coreset_samples):
            # anchor 값이 가장 큰 패치를 코어셋으로 선택한다.
            select_idx = torch.argmax(coreset_anchor_distances).item()
            coreset_indices.append(select_idx)
            
            # 모든 포인트와 선택된 코어셋 포인트 간의 거리가 담긴 벡터를 가져온다
            coreset_select_distance = distance_matrix[
                :, select_idx : select_idx + 1  # noqa E203
            ]
            
            # anchor 갱신. 선택된 코어셋 포인트 간의 거리가 여러개가 가능함. 왜냐하면 코어셋 포인트 자체가 여러개이므로.
            # 하지만 그 여러개의 거리를 다 저장하는 것이 아니라, 최소값만 저장을 하면된다.
            # 이 때, 새로 뽑힌 코어셋 포인트를 반영해서 최소값을 갱신하게 된다.
            
            
            '''
coreset_select_distance: 이 텐서는 최근에 선택된 코어셋 포인트와 다른 모든 포인트들 사이의 거리를 담고 있습니다. 
이 거리는 새로운 후보로 고려되며, [N, 1] 형태를 가집니다.
torch.cat: 이 함수는 주어진 차원(dim)을 따라 여러 텐서를 연결합니다. 여기서 dim=1는 열 방향으로 텐서들을 결합하라는 의미입니다.
[coreset_anchor_distances.unsqueeze(-1), coreset_select_distance]: unsqueeze로 차원이 조정된 coreset_anchor_distances와 coreset_select_distance를 열 방향으로 결합합니다. 
결과적으로, 각 데이터 포인트에 대해 두 개의 거리 값(기존 최소 거리와 새로운 후보 거리)을 가진 새로운 텐서가 생성됩니다. 즉, 결과 텐서의 모양은 [N, 2]가 됩니다.
생성된 [N, 2] 형태의 텐서는 후속 처리에서 각 포인트의 새로운 최소 거리를 결정하기 위해 사용됩니다. 이 과정에서 각 행의 두 거리 값 중 더 작은 값이 선택되어 해당 포인트의 최종 최소 거리로 업데이트됩니다.
이러한 방식으로 Greedy Coreset 알고리즘은 각 단계에서 새로운 코어셋 포인트를 추가할 때마다 데이터 세트 전체의 거리 정보를 갱신하고, 각 포인트의 최소 거리를 점진적으로 갱신하여 전체 데이터 세트를 효과적으로 요약합니다.
            '''
            coreset_anchor_distances = torch.cat(
                [coreset_anchor_distances.unsqueeze(-1), coreset_select_distance], dim=1
            )
            coreset_anchor_distances = torch.min(coreset_anchor_distances, dim=1).values

        return np.array(coreset_indices)


class ApproximateGreedyCoresetSampler(GreedyCoresetSampler):
    def __init__(
        self,
        percentage: float,
        device: torch.device,
        number_of_starting_points: int = 10,
        dimension_to_project_features_to: int = 128,
    ):
        """Approximate Greedy Coreset sampling base class."""
        self.number_of_starting_points = number_of_starting_points
        super().__init__(percentage, device, dimension_to_project_features_to)

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        """Runs approximate iterative greedy coreset selection.
        This greedy coreset implementation does not require computation of the
        full N x N distance matrix and thus requires a lot less memory, however
        at the cost of increased sampling times.
        Args:
            features: [NxD] input feature bank to sample.
        """
        number_of_starting_points = np.clip(
            self.number_of_starting_points, None, len(features)
        )
        start_points = np.random.choice(
            len(features), number_of_starting_points, replace=False
        ).tolist()

        approximate_distance_matrix = self._compute_batchwise_differences(
            features, features[start_points]
        )
        approximate_coreset_anchor_distances = torch.mean(
            approximate_distance_matrix, axis=-1
        ).reshape(-1, 1)
        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percentage)

        with torch.no_grad():
            for _ in tqdm.tqdm(range(num_coreset_samples), desc="Subsampling..."):
                select_idx = torch.argmax(approximate_coreset_anchor_distances).item()
                coreset_indices.append(select_idx)
                coreset_select_distance = self._compute_batchwise_differences(
                    features, features[select_idx : select_idx + 1]  # noqa: E203
                )
                approximate_coreset_anchor_distances = torch.cat(
                    [approximate_coreset_anchor_distances, coreset_select_distance],
                    dim=-1,
                )
                approximate_coreset_anchor_distances = torch.min(
                    approximate_coreset_anchor_distances, dim=1
                ).values.reshape(-1, 1)

        return np.array(coreset_indices)


class RandomSampler(BaseSampler):
    def __init__(self, percentage: float):
        super().__init__(percentage)

    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Randomly samples input feature collection.
        Args:
            features: [N x D]
        """
        num_random_samples = int(len(features) * self.percentage)
        subset_indices = np.random.choice(
            len(features), num_random_samples, replace=False
        )
        subset_indices = np.array(subset_indices)
        return features[subset_indices]