"""OmniGraph node simulating the RH8D's fingertip force sensors (Seed Robotics FTS3).

On the robot, each RH8D fingertip carries a 3D force sensor; third_party/ros2_fts
(script/read_publish_sensor_node.py, launched by nicol_controller's nicol_sensor.launch.py in the
namespaces right / left) publishes ros2_sensor_pkg/AllSensors on /<side>/AllSensors at 50 Hz: header
(stamp, empty frame_id), length = number of sensors, data = one LoneSensor per present sensor with id,
fx, fy, fz (int64), abs, yaw, pitch, is_present, is_3d. Right hand 5 sensors (thumb, index, middle,
ring, little) on both hands (the left middle finger's experimental Optoforce was replaced by the FTS3,
2026-11-15).

Here the force of a tip is the net force of all its contacts (PhysX contact report; the tip bodies
need PhysxContactReportAPI, nicol_description usd/robots/nicol_overrides/nicol_contact_report.usda)
in that tip's frame - the observation nicol_isaaclab's lift policy is trained on
(mdp.fingertip_forces_local, normal forces only) - times forceScale (1000: N -> mN, the scale both
Gazebo NicolF3DPlugins apply "like the real sensor output"). What could not be established: the FTS3's
own axes (sensor pad frame) and unit from a datasheet; the hardware's sign convention.

The contact report comes per physics step (subscribe_physics_step_events + get_contact_report). PhysX
reports an impulse on actor0 of a pair; a pair is kept from CONTACT_FOUND until CONTACT_LOST, so a
grasp whose bodies fall asleep (no more reports) keeps its last force.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np


def _resolve_domain_id(db) -> int | None:
    """Domain ID of the ROS2 Context node connected to inputs:context, using the same rules as that node.

    Returns None if nothing is connected, which lets rclpy fall back to $ROS_DOMAIN_ID or 0.
    """
    upstream = db.node.get_attribute("inputs:context").get_upstream_connections()
    if not upstream:
        return None
    ctx_node = upstream[0].get_node()
    if not ctx_node.get_type_name().endswith("ROS2Context"):
        raise ValueError(f"inputs:context must be connected to a ROS2 Context node, got {ctx_node.get_type_name()}")
    if ctx_node.get_attribute("inputs:useDomainIDEnvVar").get() and os.environ.get("ROS_DOMAIN_ID"):
        return int(os.environ["ROS_DOMAIN_ID"], 0)
    return int(ctx_node.get_attribute("inputs:domain_id").get())


def _qos_from_json(qos_json: str) -> object:
    """Convert the JSON written by the ROS2 QoS Profile node to an rclpy QoSProfile; empty means depth 10, volatile."""
    from rclpy.duration import Duration
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, LivelinessPolicy, QoSProfile, ReliabilityPolicy

    if not qos_json.strip():
        return QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
    # token names as mapped in isaacsim.ros2.core Ros2QoS.hpp
    history = {"systemDefault": HistoryPolicy.SYSTEM_DEFAULT, "keepLast": HistoryPolicy.KEEP_LAST,
               "keepAll": HistoryPolicy.KEEP_ALL, "unknown": HistoryPolicy.UNKNOWN}
    reliability = {"systemDefault": ReliabilityPolicy.SYSTEM_DEFAULT, "reliable": ReliabilityPolicy.RELIABLE,
                   "bestEffort": ReliabilityPolicy.BEST_EFFORT, "unknown": ReliabilityPolicy.UNKNOWN}
    durability = {"systemDefault": DurabilityPolicy.SYSTEM_DEFAULT, "transientLocal": DurabilityPolicy.TRANSIENT_LOCAL,
                  "volatile": DurabilityPolicy.VOLATILE, "unknown": DurabilityPolicy.UNKNOWN}
    liveliness = {"systemDefault": LivelinessPolicy.SYSTEM_DEFAULT, "automatic": LivelinessPolicy.AUTOMATIC,
                  "manualByTopic": LivelinessPolicy.MANUAL_BY_TOPIC, "unknown": LivelinessPolicy.UNKNOWN}
    q = json.loads(qos_json)
    return QoSProfile(
        history=history[q["history"]],
        depth=int(q["depth"]),
        reliability=reliability[q["reliability"]],
        durability=durability[q["durability"]],
        deadline=Duration(seconds=q["deadline"]),
        lifespan=Duration(seconds=q["lifespan"]),
        liveliness=liveliness[q["liveliness"]],
        liveliness_lease_duration=Duration(seconds=q["leaseDuration"]),
    )


def _rotate_inverse(q_xyzw, v: np.ndarray) -> np.ndarray:
    """Rotate v by the inverse of the unit quaternion q (x, y, z, w): world -> body frame."""
    x, y, z, w = q_xyzw
    u = -np.array([x, y, z])
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def ros2_fts_polar(fx: float, fy: float, fz: float) -> tuple[float, float, float]:
    """abs, yaw, pitch of a reading as ros2_fts computes them (cart2sph): pitch = elevation, yaw = azimuth."""
    hxy = math.hypot(fx, fy)
    return math.hypot(hxy, fz), math.atan2(fy, fx), math.pi / 2 - math.atan2(hxy, fz)


class OgnROS2PublishFingertipForcesInternalState:
    """Per-node state: the ROS 2 publisher on its own rclpy context and the contact pairs of the tips."""

    def __init__(self) -> None:
        self.ros_key = None
        self.failed_key = None
        self.context = None
        self.node = None
        self.publisher = None
        self.tips = []              # tip body paths, sensor id order
        self.tip_ids = {}           # sdfPathToInt(tip) -> index
        self.include_friction = False
        self.pairs = {}             # (tip index, collider0, collider1) -> world force on the tip [N]
        self.step_sub = None
        self.next_publish = None
        self.last_time = None

    # contact report, after every physics step (main thread)
    def on_physics_step(self, dt: float) -> None:
        from omni.physx import get_physx_simulation_interface
        from omni.physx.bindings._physx import ContactEventType

        if dt <= 0.0 or not self.tip_ids:
            return
        sim = get_physx_simulation_interface()
        if self.include_friction:
            headers, data, anchors = sim.get_full_contact_report()
        else:
            (headers, data), anchors = sim.get_contact_report(), None
        for h in headers:
            for actor, sign in ((h.actor0, 1.0), (h.actor1, -1.0)):
                tip = self.tip_ids.get(actor)
                if tip is None:
                    continue
                key = (tip, h.collider0, h.collider1)
                if h.type == ContactEventType.CONTACT_LOST:
                    self.pairs.pop(key, None)
                    continue
                # the impulse acts on actor0, normal and friction alike (measured: a 1 kg box resting on a 20 deg
                # slope, actor0: normal 9.22 N along the slope normal + friction 3.36 N up the slope = (0, 0, 9.81) N)
                imp = np.zeros(3)
                for i in range(h.contact_data_offset, h.contact_data_offset + h.num_contact_data):
                    d = data[i].impulse
                    imp += (d[0], d[1], d[2])
                if anchors is not None:
                    for i in range(h.friction_anchors_offset, h.friction_anchors_offset + h.num_friction_anchors_data):
                        d = anchors[i].impulse
                        imp += (d[0], d[1], d[2])
                self.pairs[key] = sign * imp / dt

    def start(self, tips: list[str], include_friction: bool) -> None:
        """Track the contacts of the tip bodies."""
        from omni.physx import get_physx_interface
        from pxr import PhysicsSchemaTools, Sdf

        self.stop()
        self.tips = tips
        self.tip_ids = {PhysicsSchemaTools.sdfPathToInt(Sdf.Path(p)): i for i, p in enumerate(tips)}
        self.include_friction = include_friction
        self.step_sub = get_physx_interface().subscribe_physics_step_events(self.on_physics_step)

    def stop(self) -> None:
        self.step_sub = None
        self.pairs.clear()
        self.tips, self.tip_ids = [], {}

    def tip_forces(self) -> np.ndarray:
        """(n, 3) net contact force of each tip in its own frame [N]."""
        from omni.physx import get_physx_interface

        f_w = np.zeros((len(self.tips), 3))
        for (tip, _, _), f in self.pairs.items():
            f_w[tip] += f
        physx = get_physx_interface()
        out = np.zeros_like(f_w)
        for i, path in enumerate(self.tips):
            if not f_w[i].any():
                continue
            tf = physx.get_rigidbody_transformation(path)
            r = tf["rotation"]
            out[i] = _rotate_inverse((r[0], r[1], r[2], r[3]), f_w[i]) if tf["ret_val"] else f_w[i]
        return out

    def start_ros(self, node_name: str, namespace: str, topic: str, domain_id: int | None, qos: object) -> None:
        import rclpy
        from ros2_sensor_pkg.msg import AllSensors

        # own context, so this node neither depends on nor interferes with other rclpy users in the process
        self.context = rclpy.Context()
        # args=[]: without it rclpy uses sys.argv, i.e. the host process' --ros-args
        self.context.init(args=[], domain_id=domain_id)
        self.node = rclpy.create_node(
            node_name, namespace=namespace or None, context=self.context, use_global_arguments=False
        )
        self.publisher = self.node.create_publisher(AllSensors, topic, qos)

    def shutdown_ros(self) -> None:
        if self.node is not None:
            self.node.destroy_node()
        if self.context is not None:
            self.context.try_shutdown()
        self.context = self.node = self.publisher = None
        self.ros_key = None

    def due(self, t: float, rate: float) -> bool:
        """True if a message is due at simulation time t; a time jump back (stop/play) restarts the clock."""
        if self.last_time is not None and t < self.last_time:
            self.next_publish = None
            self.pairs.clear()
        self.last_time = t
        if rate <= 0.0:
            return True
        if self.next_publish is None or t >= self.next_publish - 1e-9:
            period = 1.0 / rate
            self.next_publish = t + period if self.next_publish is None else max(self.next_publish + period, t)
            return True
        return False

    def publish(self, forces: np.ndarray, scale: float, stamp: float, frame_id: str) -> None:
        from ros2_sensor_pkg.msg import AllSensors, LoneSensor

        msg = AllSensors()
        msg.header.stamp.sec = int(math.floor(stamp))
        msg.header.stamp.nanosec = int((stamp - math.floor(stamp)) * 1e9)
        msg.header.frame_id = frame_id
        msg.length = len(forces)
        for i, f in enumerate(forces):
            s = LoneSensor()
            s.id = i
            s.fx, s.fy, s.fz = (int(round(v * scale)) for v in f)
            s.abs, s.yaw, s.pitch = ros2_fts_polar(float(s.fx), float(s.fy), float(s.fz))
            s.is_present = True
            s.is_3d = True
            msg.data.append(s)
        self.publisher.publish(msg)


class OgnROS2PublishFingertipForces:
    """OmniGraph node that publishes the fingertip contact forces as ros2_sensor_pkg/AllSensors."""

    @staticmethod
    def internal_state() -> OgnROS2PublishFingertipForcesInternalState:
        return OgnROS2PublishFingertipForcesInternalState()

    @staticmethod
    def compute(db) -> bool:
        import omni.graph.core as og

        state = db.per_instance_state
        db.outputs.execOut = og.ExecutionAttributeState.DISABLED
        # ticks that publish nothing (the graph ticks once per physics step, 60 Hz; 50 Hz skips 1 of 6) return
        # before reading the other inputs; input changes take effect at the next message
        stamp = db.inputs.timeStamp
        if not state.due(stamp, db.inputs.publishRate):
            return True
        try:
            domain_id = _resolve_domain_id(db)
        except Exception as exc:
            db.log_error(f"Failed to resolve ROS 2 domain ID: {exc}")
            return False
        tips = [str(p) for p in db.inputs.tipPrims]
        ros_key = (db.inputs.nodeName, db.inputs.nodeNamespace, db.inputs.topicName, domain_id, db.inputs.qosProfile,
                   tuple(tips), db.inputs.includeFriction)

        # a failure is latched until an input changes, so errors are not repeated every tick
        if ros_key == state.failed_key:
            return False
        if ros_key != state.ros_key:
            state.shutdown_ros()
            try:
                if not tips:
                    raise ValueError("inputs:tipPrims is empty")
                state.start(tips, db.inputs.includeFriction)
                state.start_ros(db.inputs.nodeName, db.inputs.nodeNamespace, db.inputs.topicName, domain_id,
                                _qos_from_json(db.inputs.qosProfile))
            except Exception as exc:
                db.log_error(f"Failed to create the fingertip force publisher: {exc}")
                state.shutdown_ros()
                state.stop()
                state.failed_key = ros_key
                return False
            state.ros_key = ros_key
        state.failed_key = None

        forces = state.tip_forces()
        db.outputs.forces = forces
        state.publish(forces, db.inputs.forceScale, stamp, db.inputs.frameId)
        db.outputs.execOut = og.ExecutionAttributeState.ENABLED
        return True

    @staticmethod
    def release_instance(node, graph_instance_id: int) -> None:
        """Shut down the ROS 2 node and the contact subscription when the node instance is removed."""
        from isaacsim.nicol.nodes.ogn.OgnROS2PublishFingertipForcesDatabase import (
            OgnROS2PublishFingertipForcesDatabase,
        )

        try:
            state = OgnROS2PublishFingertipForcesDatabase.get_internal_state(node, graph_instance_id)
        except Exception:
            state = None
        if state is not None:
            state.shutdown_ros()
            state.stop()
