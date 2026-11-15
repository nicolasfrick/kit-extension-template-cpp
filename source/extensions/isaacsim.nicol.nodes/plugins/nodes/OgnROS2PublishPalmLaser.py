"""OmniGraph node simulating the RH8D's palm time-of-flight range sensor.

On the robot, the RH8D palm carries an IR range sensor (register Palm_IR_Sensor, read by nicol_libs'
SensorDynamixel); nicol_controller publishes it as nicol_msgs/PalmLaser (header, float64 data) on
/<side>/laser, in mm (nicol_gui shows it as is; the Gazebo NicolLaserPlugin publishes round(range * 1000)
with a 3-254 mm lidar at 50 Hz). The sensor frame is the URDF's r_laser / l_laser: palm xyz
(-0.0003, 0.0450, 0), rpy (0, 0, pi), the beam along its +x (the palm's -x, the direction the palm faces).

Here one PhysX scene-query ray (raycast_all) from the sensor prim's origin along its +x, up to maxRange
(2-200 mm per the user's spec of the sensor); maxRange when nothing is nearer, minRange for nearer hits.
The ray starts inside the palm's collider: hits on the sensor's own body are skipped (ignoreHand: the whole
hand's). The same range the lift policy observes (nicol_isaaclab mdp.palm_tof_distance), except that
Isaac Lab's ray only sees the object's bounding box and the plate, this one every collider.
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


def _rotate(q_xyzw, v: np.ndarray) -> np.ndarray:
    """Rotate v by the unit quaternion q (x, y, z, w): body -> world frame."""
    x, y, z, w = q_xyzw
    u = np.array([x, y, z])
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


class OgnROS2PublishPalmLaserInternalState:
    """Per-node state: the ROS 2 publisher on its own rclpy context and the sensor's mount on its body."""

    def __init__(self) -> None:
        self.ros_key = None
        self.failed_key = None
        self.context = None
        self.node = None
        self.publisher = None
        self.body = ""              # rigid body carrying the sensor (the palm)
        self.ignore_prefix = ""     # skip hits on bodies below this path (ignoreHand)
        self.offset = np.zeros(3)   # sensor origin in the body frame
        self.axis = np.array([1.0, 0.0, 0.0])   # sensor +x in the body frame
        self.next_publish = None
        self.last_time = None

    def mount(self, sensor: str, ignore_hand: bool) -> None:
        """Find the sensor's rigid body and its fixed pose on it (static: authored xformOps)."""
        import omni.usd
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(sensor)
        if not prim or not prim.IsValid():
            raise ValueError(f"sensor prim {sensor} not found")
        body = prim
        while body and body.IsValid() and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body = body.GetParent()
        if not body or not body.IsValid() or body.IsPseudoRoot():
            raise ValueError(f"no rigid body above {sensor}")
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        rel = cache.GetLocalToWorldTransform(prim) * cache.GetLocalToWorldTransform(body).GetInverse()
        self.offset = np.array(rel.ExtractTranslation())
        self.axis = np.array(rel.TransformDir(Gf.Vec3d(1, 0, 0)).GetNormalized())
        self.body = str(body.GetPath())
        self.ignore_prefix = str(body.GetParent().GetPath()) + "/" if ignore_hand else ""

    def measure(self, min_range: float, max_range: float) -> tuple[float, str]:
        """Range [m] clipped to [min_range, max_range] and the collider hit within max_range ('' if none)."""
        import carb
        from omni.physx import get_physx_interface, get_physx_scene_query_interface

        tf = get_physx_interface().get_rigidbody_transformation(self.body)
        if not tf["ret_val"]:
            return max_range, ""
        p, r = tf["position"], tf["rotation"]
        q = (r[0], r[1], r[2], r[3])
        origin = np.array([p[0], p[1], p[2]]) + _rotate(q, self.offset)
        direction = _rotate(q, self.axis)
        best = [math.inf, ""]

        def report(hit) -> bool:
            body = hit.rigid_body
            if body == self.body or (self.ignore_prefix and body.startswith(self.ignore_prefix)):
                return True
            if hit.distance < best[0]:
                best[0], best[1] = hit.distance, hit.collision
            return True

        get_physx_scene_query_interface().raycast_all(
            carb.Float3(*origin), carb.Float3(*direction), max_range, report)
        if best[0] > max_range:
            return max_range, ""
        return max(best[0], min_range), best[1]

    def start_ros(self, node_name: str, namespace: str, topic: str, domain_id: int | None, qos: object) -> None:
        import rclpy
        from nicol_msgs.msg import PalmLaser

        # own context, so this node neither depends on nor interferes with other rclpy users in the process
        self.context = rclpy.Context()
        # args=[]: without it rclpy uses sys.argv, i.e. the host process' --ros-args
        self.context.init(args=[], domain_id=domain_id)
        self.node = rclpy.create_node(
            node_name, namespace=namespace or None, context=self.context, use_global_arguments=False
        )
        self.publisher = self.node.create_publisher(PalmLaser, topic, qos)

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
        self.last_time = t
        if rate <= 0.0:
            return True
        if self.next_publish is None or t >= self.next_publish - 1e-9:
            period = 1.0 / rate
            self.next_publish = t + period if self.next_publish is None else max(self.next_publish + period, t)
            return True
        return False

    def publish(self, value: float, stamp: float, frame_id: str) -> None:
        from nicol_msgs.msg import PalmLaser

        msg = PalmLaser()
        msg.header.stamp.sec = int(math.floor(stamp))
        msg.header.stamp.nanosec = int((stamp - math.floor(stamp)) * 1e9)
        msg.header.frame_id = frame_id
        msg.data = float(value)
        self.publisher.publish(msg)


class OgnROS2PublishPalmLaser:
    """OmniGraph node that publishes the palm's TOF range as nicol_msgs/PalmLaser."""

    @staticmethod
    def internal_state() -> OgnROS2PublishPalmLaserInternalState:
        return OgnROS2PublishPalmLaserInternalState()

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
        sensor = str(db.inputs.sensorPrim[0]) if len(db.inputs.sensorPrim) else ""
        ros_key = (db.inputs.nodeName, db.inputs.nodeNamespace, db.inputs.topicName, domain_id, db.inputs.qosProfile,
                   sensor, db.inputs.ignoreHand)

        # a failure is latched until an input changes, so errors are not repeated every tick
        if ros_key == state.failed_key:
            return False
        if ros_key != state.ros_key:
            state.shutdown_ros()
            try:
                if not sensor:
                    raise ValueError("inputs:sensorPrim is not set")
                state.mount(sensor, db.inputs.ignoreHand)
                state.start_ros(db.inputs.nodeName, db.inputs.nodeNamespace, db.inputs.topicName, domain_id,
                                _qos_from_json(db.inputs.qosProfile))
            except Exception as exc:
                db.log_error(f"Failed to create the palm laser publisher: {exc}")
                state.shutdown_ros()
                state.failed_key = ros_key
                return False
            state.ros_key = ros_key
        state.failed_key = None

        rng, hit = state.measure(db.inputs.minRange, db.inputs.maxRange)
        db.outputs.range = rng
        db.outputs.hitPrim = hit
        value = rng * db.inputs.rangeScale
        state.publish(round(value) if db.inputs.roundOutput else value, stamp, db.inputs.frameId)
        db.outputs.execOut = og.ExecutionAttributeState.ENABLED
        return True

    @staticmethod
    def release_instance(node, graph_instance_id: int) -> None:
        """Shut down the ROS 2 node when the node instance is removed."""
        from isaacsim.nicol.nodes.ogn.OgnROS2PublishPalmLaserDatabase import OgnROS2PublishPalmLaserDatabase

        try:
            state = OgnROS2PublishPalmLaserDatabase.get_internal_state(node, graph_instance_id)
        except Exception:
            state = None
        if state is not None:
            state.shutdown_ros()
