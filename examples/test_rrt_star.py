"""
    Demonstrates how to use the RRT* algorithm with the colav-simulator. Note that the underlying ship model in the planner does not
    match the ship model used from the colav-simulator, and thus should be tuned for usage outside this example.

    Remember to install the dependencies (COLAV-simulator) before running this script.

    Author: Trym Tengesdal
"""

import pathlib
import time
from dataclasses import asdict, dataclass, field
from typing import Optional, Tuple

import colav_simulator.common.map_functions as mapf
import colav_simulator.common.paths as dp
import colav_simulator.common.plotters as plotters
import colav_simulator.core.colav.colav_interface as ci
import colav_simulator.core.guidances as guidances
import colav_simulator.core.models as models
import colav_simulator.core.stochasticity as stochasticity
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rrt_star_lib
import seacharts.enc as senc
from colav_simulator.scenario_generator import ScenarioGenerator
from colav_simulator.simulator import Simulator
from shapely import strtree


def parse_rrt_solution(soln: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Parses the RRT solution.

    Args:
        soln (dict): Solution dictionary.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]: Tuple of waypoints, trajectory, inputs and times and cost from the solution.
    """
    times = np.array(soln["times"])
    n_samples = len(times)
    trajectory = np.zeros((6, n_samples))
    n_inputs = len(soln["inputs"])
    inputs = np.zeros((3, n_inputs))
    n_wps = len(soln["waypoints"])
    waypoints = np.zeros((3, n_wps))
    if n_samples > 0:
        for k in range(n_wps):
            waypoints[:, k] = np.array(soln["waypoints"][k])
        for k in range(n_samples):
            trajectory[:, k] = np.array(soln["states"][k])
        for k in range(n_inputs):
            inputs[:, k] = np.array(soln["inputs"][k])
    if n_wps == 1:
        waypoints = np.array([waypoints[:, 0], waypoints[:, 0]]).T
    return waypoints, trajectory, inputs, times, soln["cost"]


@dataclass
class RRTStarParams:
    max_nodes: int = 3000
    max_iter: int = 25000
    max_time: float = 5000.0
    iter_between_direct_goal_growth: int = 250
    min_node_dist: float = 10.0
    goal_radius: float = 700.0
    step_size: float = 0.5
    min_steering_time: float = 5.0
    max_steering_time: float = 20.0
    steering_acceptance_radius: float = 10.0
    gamma: float = 1200.0

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = cls(**config_dict)
        return config

    def to_dict(self):
        return asdict(self)


@dataclass
class RRTConfig:
    params: RRTStarParams = field(default_factory=lambda: RRTStarParams())
    model: models.KinematicCSOGParams = field(
        default_factory=lambda: models.KinematicCSOGParams(
            name="KinematicCSOG",
            draft=0.5,
            length=15.0,
            width=4.0,
            T_chi=6.0,
            T_U=6.0,
            r_max=np.deg2rad(10),
            U_min=0.0,
            U_max=10.0,
        )
    )
    los: guidances.LOSGuidanceParams = field(
        default_factory=lambda: guidances.LOSGuidanceParams(
            K_p=0.035,
            K_i=0.0,
            pass_angle_threshold=90.0,
            R_a=25.0,
            max_cross_track_error_int=30.0,
            cross_track_error_int_threshold=30.0,
        )
    )

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = RRTConfig(
            params=RRTStarParams.from_dict(config_dict["params"]),
            model=models.KinematicCSOGParams.from_dict(config_dict["model"]),
            los=guidances.LOSGuidanceParams.from_dict(config_dict["los"]),
        )

        return config


@dataclass
class RRTPlannerParams:
    los: guidances.LOSGuidanceParams = field(
        default_factory=lambda: guidances.LOSGuidanceParams(
            K_p=0.035, K_i=0.0, pass_angle_threshold=90.0, R_a=25.0, max_cross_track_error_int=30.0
        )
    )
    rrt: RRTConfig = field(default_factory=lambda: RRTConfig())

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = RRTPlannerParams(
            los=guidances.LOSGuidanceParams.from_dict(config_dict["los"]),
            rrt=RRTConfig.from_dict(config_dict["pq-rrt"]),
        )

        return config


class RRTStar(ci.ICOLAV):
    def __init__(self, config: RRTPlannerParams) -> None:
        self._rrt_config = config.rrt
        self._rrt = rrt_star_lib.RRTStar(config.rrt.los, config.rrt.model, config.rrt.params)
        self._los = guidances.LOSGuidance(config.los)

        self._rrt_inputs: np.ndarray = np.empty(3)
        self._rrt_trajectory: np.ndarray = np.empty(6)
        self._rrt_waypoints: np.ndarray = np.empty(3)
        self._geometry_tree: strtree.STRtree = strtree.STRtree([])

        self._min_depth: int = 0

        self._map_origin: np.ndarray = np.array([])
        self._references = np.array([])
        self._initialized = False
        self._t_prev: float = 0.0

    def plan(
        self,
        t: float,
        waypoints: np.ndarray,
        speed_plan: np.ndarray,
        ownship_state: np.ndarray,
        do_list: list,
        enc: Optional[senc.ENC] = None,
        goal_state: Optional[np.ndarray] = None,
        w: Optional[stochasticity.DisturbanceData] = None,
        **kwargs,
    ) -> np.ndarray:
        assert goal_state is not None, "Goal state must be provided to the RRT"
        assert enc is not None, "ENC must be provided to the RRT"
        if t == 0:
            self.reset()
        if not self._initialized:
            self._min_depth = 0  # mapf.find_minimum_depth(kwargs["os_draft"], enc)
            self._t_prev = t
            self._map_origin = ownship_state[:2]
            self._initialized = True
            relevant_grounding_hazards = mapf.extract_relevant_grounding_hazards_as_union(
                self._min_depth, enc, buffer=5.0, show_plots=True
            )
            self._geometry_tree, _ = mapf.fill_rtree_with_geometries(relevant_grounding_hazards)
            safe_sea_triangulation = mapf.create_safe_sea_triangulation(enc, self._min_depth, show_plots=False)
            self._rrt.transfer_bbox(enc.bbox)
            self._rrt.transfer_enc_hazards(relevant_grounding_hazards[0])
            self._rrt.transfer_safe_sea_triangulation(safe_sea_triangulation)
            self._rrt.set_goal_state(goal_state.tolist())

            U_d = ownship_state[3]  # Constant desired speed given by the initial own-ship speed
            rrt_solution: dict = self._rrt.grow_towards_goal(
                ownship_state=ownship_state.tolist(),
                U_d=U_d,
                initialized=False,
                return_on_first_solution=False,
                verbose=True,
            )
            self._rrt_waypoints, self._rrt_trajectory, self._rrt_inputs, times, cost = parse_rrt_solution(rrt_solution)

            if enc is not None:
                plotters.plot_rrt_tree(self._rrt.get_tree_as_list_of_dicts(), enc)
                plotters.plot_waypoints(
                    self._rrt_waypoints,
                    enc,
                    color="orange",
                    point_buffer=3.0,
                    disk_buffer=6.0,
                    hole_buffer=3.0,
                )
                ship_poly = mapf.create_ship_polygon(
                    ownship_state[0],
                    ownship_state[1],
                    ownship_state[2],
                    kwargs["os_length"],
                    kwargs["os_width"],
                    2.0,
                    2.0,
                )
                print(f"Num tree nodes: {self._rrt.get_num_nodes()}")
                enc.draw_polygon(ship_poly, color="yellow")
                enc.draw_circle(center=(goal_state[1], goal_state[0]), radius=20.0, color="magenta", alpha=0.7)
        else:
            self._rrt_trajectory = self._rrt_trajectory[:, 1:]
            self._rrt_inputs = self._rrt_inputs[:, 1:]

        self._t_prev = t
        self._references = self._los.compute_references(
            self._rrt_waypoints[:2, :],
            speed_plan=self._rrt_waypoints[2, :],
            times=None,
            xs=ownship_state,
            dt=t - self._t_prev,
        )
        return self._references

    def get_current_plan(self) -> np.ndarray:
        return self._references

    def get_colav_data(self) -> dict:
        if not self._initialized:
            return {
                "nominal_trajectory": np.zeros((6, 1)),
                "nominal_inputs": np.zeros((3, 1)),
                "params": self._rrt_config.params,
                "t": self._t_prev,
            }
        else:
            return {
                "nominal_trajectory": self._rrt_trajectory,
                "nominal_inputs": self._rrt_inputs,
                "params": self._rrt_config.params,
                "t": self._t_prev,
            }

    def plot_results(self, ax_map: plt.Axes, enc: senc.ENC, plt_handles: dict, **kwargs) -> dict:
        if self._rrt_trajectory.size > 6:
            plt_handles["colav_nominal_trajectory"].set_xdata(self._rrt_trajectory[1, 0:])
            plt_handles["colav_nominal_trajectory"].set_ydata(self._rrt_trajectory[0, 0:])
        return plt_handles

    def reset(self):
        """Resets the RRTStar to its initial state."""
        self._rrt_inputs: np.ndarray = np.empty(3)
        self._rrt_trajectory: np.ndarray = np.empty(6)
        self._rrt_waypoints: np.ndarray = np.empty(3)
        self._geometry_tree: strtree.STRtree = strtree.STRtree([])

        self._min_depth: int = 0

        self._map_origin: np.ndarray = np.array([])
        self._references = np.array([])
        self._initialized = False
        self._t_prev: float = 0.0


if __name__ == "__main__":
    params = RRTPlannerParams()
    scenario_file = dp.scenarios / "rrt_test.yaml"

    params.rrt.params = RRTStarParams(
        max_nodes=10000,
        max_iter=25000,
        max_time=5.0,
        iter_between_direct_goal_growth=500,
        min_node_dist=5.0,
        goal_radius=100.0,
        step_size=0.5,
        min_steering_time=1.0,
        max_steering_time=30.0,
        steering_acceptance_radius=10.0,
        gamma=2000.0,
    )

    rrt = RRTStar(params)

    scenario_generator = ScenarioGenerator()
    scenario_data = scenario_generator.generate(config_file=scenario_file, new_load_of_map_data=True)
    simulator = Simulator()
    # Hint: Close ENC Seacharts plot with RRT tree to speed up livesim
    output = simulator.run([scenario_data], colav_systems=[(0, rrt)], terminate_on_collision_or_grounding=False)
    print("done")
