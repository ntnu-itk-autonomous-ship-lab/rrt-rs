"""
    Demonstrates how to use the Informed-RRT* algorithm with the colav-simulator.

    Remember to install the dependencies (COLAV-simulator) before running this script.

    Author: Trym Tengesdal
"""
import time
from dataclasses import asdict, dataclass
from typing import Optional

import colav_simulator.common.map_functions as mapf
import colav_simulator.common.paths as dp
import colav_simulator.core.colav.colav_interface as ci
import colav_simulator.core.guidances as guidances
import colav_simulator.core.models as models
import colav_simulator.core.stochasticity as stochasticity
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rrt_star_lib
import seacharts.enc as senc
from colav_simulator.scenario_management import ScenarioGenerator
from colav_simulator.simulator import Simulator
from shapely import strtree


@dataclass
class InformedRRTStarParams:
    max_nodes: int = 3000
    max_iter: int = 30000
    max_time: float = 5000.0
    iter_between_direct_goal_growth: int = 250
    min_node_dist: float = 10.0
    goal_radius: float = 700.0
    step_size: float = 0.5
    min_steering_time: float = 5.0
    max_steering_time: float = 20.0
    steering_acceptance_radius: float = 15.0
    max_nn_node_dist: float = 100.0
    gamma: float = 1200.0

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = cls(**config_dict)
        return config

    def to_dict(self):
        return asdict(self)


@dataclass
class RRTConfig:
    params: InformedRRTStarParams = InformedRRTStarParams()
    model: models.KinematicCSOGParams = models.KinematicCSOGParams(
        name="KinematicCSOG",
        draft=0.5,
        length=10.0,
        width=3.0,
        T_chi=10.0,
        T_U=7.0,
        r_max=np.deg2rad(4),
        U_min=0.0,
        U_max=15.0,
    )
    los: guidances.LOSGuidanceParams = guidances.LOSGuidanceParams(
        K_p=0.035, K_i=0.0, pass_angle_threshold=90.0, R_a=25.0, max_cross_track_error_int=30.0
    )

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = RRTConfig(
            params=InformedRRTStarParams.from_dict(config_dict["params"]),
            model=models.KinematicCSOGParams.from_dict(config_dict["model"]),
            los=guidances.LOSGuidanceParams.from_dict(config_dict["los"]),
        )

        return config


@dataclass
class RRTPlannerParams:
    los: guidances.LOSGuidanceParams = guidances.LOSGuidanceParams()
    rrt: RRTConfig = RRTConfig()

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = RRTPlannerParams(
            los=guidances.LOSGuidanceParams.from_dict(config_dict["los"]),
            rrt=RRTConfig.from_dict(config_dict["pq-rrt"]),
        )

        return config


class InformedRRTStar(ci.ICOLAV):
    def __init__(self, config: RRTPlannerParams) -> None:
        self._rrt_config = config.rrt
        self._rrt = rrt_star_lib.InformedRRTStar(config.rrt.los, config.rrt.model, config.rrt.params)
        self._los: guidances.LOSGuidance = guidances.LOSGuidance(config.los)

        self._rrt_inputs: np.ndarray = np.empty(3)
        self._rrt_trajectory: np.ndarray = np.empty(6)
        self._rrt_waypoints: np.ndarray = np.empty(2)
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
        if not self._initialized:
            self._min_depth = 0  # mapf.find_minimum_depth(kwargs["os_draft"], enc)
            self._t_prev = t
            self._map_origin = ownship_state[:2]
            self._initialized = True
            relevant_grounding_hazards = mapf.extract_relevant_grounding_hazards_as_union(
                self._min_depth, enc, buffer=10.0, show_plots=True
            )
            self._geometry_tree, _ = mapf.fill_rtree_with_geometries(relevant_grounding_hazards)
            safe_sea_triangulation = mapf.create_safe_sea_triangulation(enc, self._min_depth, show_plots=False)
            self._rrt.transfer_bbox(enc.bbox)
            self._rrt.transfer_enc_hazards(relevant_grounding_hazards[0])
            self._rrt.transfer_safe_sea_triangulation(safe_sea_triangulation)
            self._rrt.set_goal_state(goal_state.tolist())

            n_mc = 100
            solution_times = []
            waypoint_list = []
            trajectory_list = []
            trajectory_timespans = []
            costs = []
            inputs = []
            success_indices = []
            # lload = pd.read_json("rrt_results.json").to_dict()
            # solution_times = list(lload["solution_times"].values())
            # waypoint_list = list(lload["waypoints"].values())
            # trajectory_list = list(lload["trajectories"].values())
            # trajectory_timespans = list(lload["trajectory_timespans"].values())
            # costs = list(lload["costs"].values())
            # inputs = list(lload["inputs"].values())

            for aa in range(0, n_mc):
                print(f"Monte Carlo iteration {aa+1}/{n_mc}")
                time_now = time.time()
                self._rrt.reset(aa)

                U_d = ownship_state[3]  # Constant desired speed given by the initial own-ship speed
                try:
                    rrt_solution: dict = self._rrt.grow_towards_goal(
                        ownship_state=ownship_state.tolist(),
                        U_d=U_d,
                        do_list=[],
                        initialized=False,
                        return_on_first_solution=False,
                    )
                except Exception as e:
                    print(e)
                    continue

                time_elapsed = time.time() - time_now
                # rrt_solution = hf.load_rrt_solution()
                times = np.array(rrt_solution["times"])
                n_samples = len(times)
                if n_samples > 0:
                    # rrt_solution = hf.load_rrt_solution()
                    success_indices.append(aa)
                    costs.append(rrt_solution["cost"])

                    self._rrt_trajectory = np.zeros((6, n_samples))
                    self._rrt_inputs = np.zeros((3, n_samples - 1))
                    n_wps = len(rrt_solution["waypoints"])
                    self._rrt_waypoints = np.zeros((3, n_wps))
                    for k in range(n_wps):
                        self._rrt_waypoints[:, k] = np.array(rrt_solution["waypoints"][k])
                    for k in range(n_samples):
                        self._rrt_trajectory[:, k] = np.array(rrt_solution["states"][k])
                        if k < n_samples - 1:
                            self._rrt_inputs[:, k] = np.array(rrt_solution["inputs"][k])

                    solution_times.append(time_elapsed)
                    waypoint_list.append(self._rrt_waypoints)
                    trajectory_list.append(self._rrt_trajectory)
                    trajectory_timespans.append(times)
                    inputs.append(self._rrt_inputs)

                if aa == 0 and enc is not None:
                    mapf.plot_rrt_tree(self._rrt.get_tree_as_list_of_dicts(), enc)
                    # mapf.plot_trajectory(self._rrt_waypoints, enc, "orange")
                    mapf.plot_waypoints(
                        self._rrt_waypoints,
                        kwargs["os_draft"],
                        enc,
                        color="orange",
                        point_buffer=2.0,
                        disk_buffer=6.0,
                        hole_buffer=2.0,
                    )
                    # mapf.plot_dynamic_obstacles(do_list, enc, 100.0, self._rrt_config.params.step_size)
                    ship_poly = mapf.create_ship_polygon(
                        ownship_state[0],
                        ownship_state[1],
                        ownship_state[2],
                        kwargs["os_length"],
                        kwargs["os_width"],
                        2.0,
                        2.0,
                    )
                    enc.draw_polygon(ship_poly, color="yellow")
                    enc.draw_circle(center=(goal_state[1], goal_state[0]), radius=20.0, color="magenta", alpha=0.3)

            solution_times = np.array(solution_times)
            costs = np.array(costs)
            print(
                f"t_solve: {solution_times.mean():.2f} +/- {solution_times.std():.2f} s | t_solve (min, max): {solution_times.min():.2f}, {solution_times.max():.2f} s"
            )
            print(
                f"cost: {costs.mean():.2f} +/- {costs.std():.2f} | cost (min, max): {costs.min():.2f}, {costs.max():.2f}"
            )
            results = {
                "solution_times": solution_times,
                "waypoints": waypoint_list,
                "trajectories": trajectory_list,
                "trajectory_timespans": trajectory_timespans,
                "inputs": inputs,
                "costs": costs,
            }
            pd.DataFrame(results).to_json("informed_rrt_star_results_smaller_planning_case.json")
            # rrt_solution = hf.load_rrt_solution()
            times = np.array(rrt_solution["times"])
            n_samples = len(times)
            if n_samples == 0:
                raise RuntimeError("RRT did not find a solution")
            self._rrt_trajectory = np.zeros((6, n_samples))
            self._rrt_inputs = np.zeros((3, n_samples - 1))
            n_wps = len(rrt_solution["waypoints"])
            self._rrt_waypoints = np.zeros((3, n_wps))
            for k in range(n_wps):
                self._rrt_waypoints[:, k] = np.array(rrt_solution["waypoints"][k])
            for k in range(n_samples):
                self._rrt_trajectory[:, k] = np.array(rrt_solution["states"][k])
                if k < n_samples - 1:
                    self._rrt_inputs[:, k] = np.array(rrt_solution["inputs"][k])

            # Plots for debugging. Set breakpoint here to inspect
            if enc is not None:
                mapf.plot_rrt_tree(self._rrt.get_tree_as_list_of_dicts(), enc)
                mapf.plot_trajectory(self._rrt_waypoints, enc, "orange", marker_type="o")
                # mapf.plot_trajectory(self._rrt_trajectory, enc, "magenta")
                mapf.plot_dynamic_obstacles(do_list, enc, 100.0, self._rrt_config.params.step_size)
                ship_poly = mapf.create_ship_polygon(
                    ownship_state[0],
                    ownship_state[1],
                    ownship_state[2],
                    kwargs["os_length"],
                    kwargs["os_width"],
                    2.0,
                    2.0,
                )
                enc.draw_polygon(ship_poly, color="yellow")
                enc.draw_circle(center=(goal_state[1], goal_state[0]), radius=30.0, color="magenta", alpha=0.3)
        else:
            self._rrt_trajectory = self._rrt_trajectory[:, 1:]
            self._rrt_inputs = self._rrt_inputs[:, 1:]

        self._t_prev = t
        # Alternative 1: Use LOS-guidance to track the trajectory
        self._references = self._los.compute_references(
            self._rrt_waypoints[:2, :],
            speed_plan=self._rrt_waypoints[2, :],
            times=None,
            xs=ownship_state,
            dt=t - self._t_prev,
        )

        # Alternative 2: Apply inputs directly to the ownship (bad todo in practice)
        # self._references = np.zeros((9, len(self._rrt_inputs[0, :])))
        # self._references[:2, :] = self._rrt_inputs[:2, :]
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
            plt_handles["colav_nominal_trajectory"].set_xdata(self._rrt_trajectory[1, 0:-1:10])
            plt_handles["colav_nominal_trajectory"].set_ydata(self._rrt_trajectory[0, 0:-1:10])

        return plt_handles


if __name__ == "__main__":
    params = RRTPlannerParams()

    choice = 1
    if choice == 0:
        scenario_file = dp.scenarios / "rl_scenario.yaml"
        params.rrt.params = InformedRRTStarParams(
            max_nodes=10000,
            max_iter=25000,
            max_time=5000.0,
            iter_between_direct_goal_growth=500,
            min_node_dist=5.0,
            goal_radius=700.0,
            step_size=1.0,
            min_steering_time=2.0,
            max_steering_time=20.0,
            steering_acceptance_radius=10.0,
            max_nn_node_dist=100.0,
            gamma=1500.0,
        )

    elif choice == 1:
        scenario_file = dp.scenarios / "rogaland_random_rl.yaml"

        params.rrt.params = InformedRRTStarParams(
            max_nodes=5000,
            max_iter=12000,
            max_time=50.0,
            iter_between_direct_goal_growth=500,
            min_node_dist=5.0,
            goal_radius=100.0,
            step_size=1.0,
            min_steering_time=2.0,
            max_steering_time=20.0,
            steering_acceptance_radius=10.0,
            max_nn_node_dist=100.0,
            gamma=1500.0,
        )

    rrt = InformedRRTStar(params)

    scenario_generator = ScenarioGenerator()
    scenario_data = scenario_generator.generate(config_file=scenario_file)
    simulator = Simulator()
    output = simulator.run([scenario_data], ownship_colav_system=rrt)
    print("done")
