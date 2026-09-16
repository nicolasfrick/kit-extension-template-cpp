"""OmniGraph node publishing a URDF robot description like robot_state_publisher, without TF."""

from __future__ import annotations

import json
import os
import re
import shlex
import threading

import omni.graph.core as og

_FIND_RE = re.compile(r"^\$\(find\s+([^)\s]+)\)")
_PACKAGE_RE = re.compile(r"^package://([^/]+)")


def _resolve_package_path(path: str) -> str:
    """Resolve a leading $(find <pkg>) or package://<pkg> to the package's share directory."""
    match = _FIND_RE.match(path) or _PACKAGE_RE.match(path)
    if not match:
        return path
    from ament_index_python.packages import get_package_share_directory

    return get_package_share_directory(match.group(1)) + path[match.end() :]


def _resolve_domain_id(db: og.Database) -> int | None:
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
    """Convert the JSON written by the ROS2 QoS Profile node to an rclpy QoSProfile; empty means latched."""
    from rclpy.duration import Duration
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, LivelinessPolicy, QoSProfile, ReliabilityPolicy

    if not qos_json.strip():
        return QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
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


def _load_description(urdf_path: str, xacro_args: str, urdf_string: str) -> str:
    """Return the URDF XML from urdf_string, or from the .urdf/.xacro file at urdf_path."""
    if urdf_string.strip():
        return urdf_string

    path = os.path.expanduser(_resolve_package_path(urdf_path.strip()))
    if not path:
        raise ValueError("either urdfPath or urdfString must be set")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"file not found: {path}")

    if not path.endswith(".xacro"):
        with open(path, "r") as f:
            return f.read()

    args = shlex.split(xacro_args)
    invalid = [arg for arg in args if ":=" not in arg]
    if invalid:
        raise ValueError(f"xacroArgs must be 'name:=value' pairs, got: {invalid}")
    import xacro  # only needed for .xacro files; requires a sourced ROS 2 environment

    mappings = dict(arg.split(":=", 1) for arg in args)
    return xacro.process_file(path, mappings=mappings).toxml()


class OgnROS2PublishRobotDescriptionInternalState:
    """Per-node state: the loaded description and a ROS 2 node on its own rclpy context."""

    def __init__(self) -> None:
        self.source_key = None
        self.ros_key = None
        self.failed_key = None
        self.description = ""
        self.dirty = False
        self.context = None
        self.node = None
        self.publisher = None
        self.executor = None
        self.spin_thread = None

    def start_ros(self, node_name: str, namespace: str, topic: str, domain_id: int | None, qos: object) -> None:
        """Create the node, publisher and parameter, and spin it in a background thread."""
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from std_msgs.msg import String

        # own context, so this node neither depends on nor interferes with other rclpy users in the process
        self.context = rclpy.Context()
        self.context.init(domain_id=domain_id)
        self.node = rclpy.create_node(node_name, namespace=namespace or None, context=self.context)
        self.node.declare_parameter("robot_description", self.description)
        self.publisher = self.node.create_publisher(String, topic, qos)

        # spinning serves the parameter services (e.g. `ros2 param get <node> robot_description`)
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()

    def publish(self) -> None:
        """Update the parameter and publish the current description."""
        from rclpy.parameter import Parameter
        from std_msgs.msg import String

        self.node.set_parameters([Parameter("robot_description", Parameter.Type.STRING, self.description)])
        self.publisher.publish(String(data=self.description))

    def shutdown_ros(self) -> None:
        """Stop spinning and destroy the ROS 2 node and context."""
        if self.executor is not None:
            self.executor.shutdown()
        if self.node is not None:
            self.node.destroy_node()
        if self.context is not None:
            self.context.try_shutdown()
        if self.spin_thread is not None:
            self.spin_thread.join(timeout=1.0)
        self.context = self.node = self.publisher = self.executor = self.spin_thread = None
        self.ros_key = None


class OgnROS2PublishRobotDescription:
    """OmniGraph node that publishes the robot description once and again whenever its inputs change."""

    @staticmethod
    def internal_state() -> OgnROS2PublishRobotDescriptionInternalState:
        """Create the per-node internal state."""
        return OgnROS2PublishRobotDescriptionInternalState()

    @staticmethod
    def compute(db: og.Database) -> bool:
        """Load the description and (re)create the publisher on input changes, then publish if needed."""
        state = db.per_instance_state
        source_key = (db.inputs.urdfPath, db.inputs.xacroArgs, db.inputs.urdfString)
        try:
            domain_id = _resolve_domain_id(db)
        except Exception as exc:
            db.log_error(f"Failed to resolve ROS 2 domain ID: {exc}")
            return False
        ros_key = (db.inputs.nodeName, db.inputs.nodeNamespace, db.inputs.topicName, domain_id, db.inputs.qosProfile)

        # a failure is latched until an input changes, so errors are not repeated every tick
        if (source_key, ros_key) == state.failed_key:
            return False

        if source_key != state.source_key:
            try:
                state.description = _load_description(*source_key)
            except Exception as exc:
                db.log_error(f"Failed to load robot description: {exc}")
                state.failed_key = (source_key, ros_key)
                return False
            state.source_key = source_key
            state.dirty = True

        if ros_key != state.ros_key:
            state.shutdown_ros()
            try:
                qos = _qos_from_json(db.inputs.qosProfile)
                from rclpy.qos import DurabilityPolicy

                if qos.durability != DurabilityPolicy.TRANSIENT_LOCAL:
                    db.log_warning(
                        "qosProfile durability is not transientLocal: subscribers started after publishing "
                        "(e.g. RViz, MoveIt) will not receive the robot description"
                    )
                state.start_ros(db.inputs.nodeName, db.inputs.nodeNamespace, db.inputs.topicName, domain_id, qos)
            except Exception as exc:
                db.log_error(f"Failed to create ROS 2 publisher: {exc}")
                state.shutdown_ros()
                state.failed_key = (source_key, ros_key)
                return False
            state.ros_key = ros_key
            state.dirty = True

        if state.dirty:
            state.publish()
            state.dirty = False

        state.failed_key = None
        db.outputs.robotDescription = state.description
        db.outputs.execOut = og.ExecutionAttributeState.ENABLED
        return True

    @staticmethod
    def release_instance(node: og.Node, graph_instance_id: int) -> None:
        """Shut down the ROS 2 node when the node instance is removed."""
        from isaacsim.nicol.nodes.ogn.OgnROS2PublishRobotDescriptionDatabase import (
            OgnROS2PublishRobotDescriptionDatabase,
        )

        try:
            state = OgnROS2PublishRobotDescriptionDatabase.get_internal_state(node, graph_instance_id)
        except Exception:
            state = None
        if state is not None:
            state.shutdown_ros()
