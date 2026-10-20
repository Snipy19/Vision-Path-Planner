import copy
import unittest

from src.algorithms.astar import AStar
from src.algorithms.bfs import BFS
from src.algorithms.bidirectional import BidirectionalAStar
from src.algorithms.dynamic_obstacles import DynamicObstacleManager
from src.evaluation.benchmark import make_scenario
from src.image_processing.grid import ImageToGrid


class CorePlannerTests(unittest.TestCase):
    def test_astar_matches_bfs_shortest_path(self):
        grid, start, goal = make_scenario(17, size=24)
        bfs_path, _ = BFS(grid).search(start, goal)
        astar_path, _ = AStar(grid, heuristic="manhattan").search(start, goal)
        self.assertIsNotNone(bfs_path)
        self.assertIsNotNone(astar_path)
        self.assertEqual(len(astar_path), len(bfs_path))

    def test_bidirectional_four_way_is_optimal(self):
        grid, start, goal = make_scenario(19, size=24)
        expected, _ = BFS(grid).search(start, goal)
        actual, _ = BidirectionalAStar(grid, allow_diagonal=False).search(start, goal)
        self.assertIsNotNone(actual)
        self.assertEqual(len(actual), len(expected))

    def test_dynamic_manager_isolated_from_input_grid(self):
        original = [[1, 1], [1, 1]]
        manager = DynamicObstacleManager(original)
        original[0][0] = 0
        manager.add_obstacle(0, 1)
        manager.reset()
        self.assertEqual(manager.get_grid(), [[1, 1], [1, 1]])
        self.assertFalse(manager.toggle(-1, 0))

    def test_scenario_is_reproducible(self):
        first = make_scenario(23, size=20)
        second = make_scenario(23, size=20)
        self.assertEqual(first, second)

    def test_image_pipeline_keeps_overlay_coordinates_aligned(self):
        grid, binary, image, gray = ImageToGrid(
            "data/maze/image1.jpg", max_dimension=32).process()
        self.assertEqual((len(grid), len(grid[0])), gray.shape)
        self.assertEqual(binary.shape, image.shape[:2])


if __name__ == "__main__":
    unittest.main()
