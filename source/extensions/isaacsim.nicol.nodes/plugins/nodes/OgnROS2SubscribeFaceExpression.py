"""OmniGraph node showing NICOL's face expression on the face's LED panels.

On the robot, LED arrays behind the translucent face shell show the eyebrows and the mouth. In the
simulation three glow panels lie on the shell (nicol_description/usd/robots/head/face_display.usda,
scripts/gen_face_display.py), meshes carrying face:region = m | l | r. nicol_face_expression publishes
one sensor_msgs/Image per region (rgba8, header.frame_id m: mouth 363x121, l / r: eyebrows 121x121) on
a latched topic whenever its FaceExpression service is called.

Each image becomes two dynamic textures (omni.ui.DynamicTextureProvider): dynamic://nicol_face_<id>
(emission, alpha = opacity) and dynamic://nicol_face_<id>_base (base color), see led_textures(). The
subscription callback converts on the ROS thread; compute (main thread) uploads, only when an expression
changed: no files, no projector, nothing per frame.

Bake an image into the panels' initial textures (same conversion; writes <out> and <out>_base), with
any Python that has numpy and OpenCV:
    python3 OgnROS2SubscribeFaceExpression.py <in.png> <out.png>
"""

from __future__ import annotations

import json
import os
import threading

import numpy as np

REGIONS = ("m", "l", "r")
TEXTURE_PREFIX = "nicol_face_"
REGION_ATTR = "face:region"

LED_FULL = 150                 # alpha of a fully lit LED in nicol_face_expression's images (ong_rgba)
LED_COLOR = (255, 145, 0)      # neon orange: the panels' color, whatever the images' (255, 150, 50)
# panel base where no LED glows, linear: the shell's 0.9 renders darker (its translucency shows the
# dark head inside), 0.82 matches it in the render (2026-11-02), so the panel's edge does not show
SHELL_GRAY = 0.82
# Transparent margin around each image, as a fraction of its width/height per side, so the blur and
# the brows (lit up to the image's side edges) can fade out. gen_face_display.py enlarges the panels
# by the same fraction: keep both PAD_FRAC equal.
PAD_FRAC = 0.2
BLUR_FRAC = 0.06               # Gaussian blur sigma, as a fraction of the image height
CORE_GAIN = 0.35               # weight of the sharp LED image in the glow
HALO_GAIN = 1.8                # weight of its blurred halo


def _gauss_blur(img: np.ndarray, sigma_y: float, sigma_x: float) -> np.ndarray:
    """Separable Gaussian blur of a 2D float array (numpy only: Kit's Python has no OpenCV)."""
    for axis, sigma in ((0, sigma_y), (1, sigma_x)):
        r = max(1, int(round(3 * sigma)))
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
        k /= k.sum()
        img = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), axis, img)
    return img


def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(np.clip(c, 0, None), 1 / 2.4) - 0.055)


def led_textures(rgba: np.ndarray, linear: bool = False, color: tuple = LED_COLOR):
    """rgba8 image of nicol_face_expression -> (emission, base color) rgba8 textures of a panel,
    PAD_FRAC larger than the image.

    The image's alpha is the LED brightness (0: off, LED_FULL: lit); the glow is the sharp
    brightness plus its blurred halo. Emission: color times the glow (premultiplied: no LED, no
    emission); its alpha is the opacity (cutout) and covers the glow and a little more. Base color:
    the shell's gray where no LED glows, turning to the LED color with the glow, so the lit core is
    saturated instead of a white surface with light on top, and the cutout edge does not show. File
    textures are decoded from sRGB by the renderer, dynamic textures are not: linear=True for those.
    """
    rgba = np.asarray(rgba, dtype=np.uint8)
    h, w = rgba.shape[:2]
    py, px = int(round(PAD_FRAC * h)), int(round(PAD_FRAC * w))
    core = np.pad(np.clip(rgba[..., 3].astype(np.float32) / LED_FULL, 0, 1), ((py, py), (px, px)))
    sigma = max(1.0, BLUR_FRAC * h)
    halo = _gauss_blur(core, sigma, sigma)
    glow = np.clip(CORE_GAIN * core + HALO_GAIN * halo, 0, 1)
    c_lin = _srgb_to_linear(np.array(color, np.float32) / 255.0)
    base_lin = SHELL_GRAY * (1 - glow[..., None]) + c_lin * glow[..., None]
    emit_lin = glow[..., None] * c_lin
    enc = (lambda x: x) if linear else _linear_to_srgb
    emission = np.zeros(core.shape + (4,), dtype=np.uint8)
    emission[..., :3] = np.round(np.clip(enc(emit_lin), 0, 1) * 255).astype(np.uint8)
    emission[..., 3] = np.where(halo > 0.01, 255, 0).astype(np.uint8)   # the glow, little beyond: the shell shows around it
    emission[[0, -1], :, 3] = 0       # filtering at the panel's edge would wrap to the other side
    emission[:, [0, -1], 3] = 0
    base = np.full(core.shape + (4,), 255, dtype=np.uint8)
    base[..., :3] = np.round(np.clip(enc(base_lin), 0, 1) * 255).astype(np.uint8)
    return np.ascontiguousarray(emission), np.ascontiguousarray(base)


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
    """Convert the JSON written by the ROS2 QoS Profile node to an rclpy QoSProfile; empty means latched, depth 10."""
    from rclpy.duration import Duration
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, LivelinessPolicy, QoSProfile, ReliabilityPolicy

    if not qos_json.strip():
        return QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
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


class OgnROS2SubscribeFaceExpressionInternalState:
    """Per-node state: a ROS 2 subscriber on its own rclpy context, the queued textures and their providers."""

    def __init__(self) -> None:
        self.ros_key = None
        self.failed_key = None
        self.context = None
        self.node = None
        self.subscription = None
        self.executor = None
        self.spin_thread = None
        self.lock = threading.Lock()
        self.pending = {}       # region -> (emission, base), only the newest image per region counts
        self.providers = {}     # region -> (emission, base) omni.ui.DynamicTextureProvider, kept alive here
        self.warnings = []      # from the ROS thread, logged in compute

    def on_image(self, msg) -> None:
        """Convert a sensor_msgs/Image (rgba8, header.frame_id m | l | r) and queue it (ROS thread)."""
        region = msg.header.frame_id
        if msg.encoding.lower() not in ("rgba8", "8uc4") or region not in REGIONS:
            with self.lock:
                self.warnings.append(f"ignored image: encoding {msg.encoding}, frame_id '{region}' (rgba8 and m | l | r expected)")
            return
        rgba = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.step)[:, :msg.width * 4]
        textures = led_textures(rgba.reshape(msg.height, msg.width, 4), linear=True)
        with self.lock:
            self.pending[region] = textures

    def start_ros(self, node_name: str, namespace: str, topic: str, domain_id: int | None, qos: object) -> None:
        """Create the node and subscription, and spin it in a background thread."""
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from sensor_msgs.msg import Image

        # own context, so this node neither depends on nor interferes with other rclpy users in the process
        self.context = rclpy.Context()
        # args=[]: without it rclpy uses sys.argv, i.e. the host process' --ros-args (e.g. a launch file's
        # __node:= remap and --params-file), which would rename this node and load foreign parameters
        self.context.init(args=[], domain_id=domain_id)
        self.node = rclpy.create_node(
            node_name, namespace=namespace or None, context=self.context, use_global_arguments=False
        )
        self.subscription = self.node.create_subscription(Image, topic, self.on_image, qos)
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()

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
        self.context = self.node = self.subscription = self.executor = self.spin_thread = None
        self.ros_key = None

    def upload(self, db) -> bool:
        """Upload the queued textures (main thread). True if a region was updated."""
        with self.lock:
            pending, self.pending = self.pending, {}
            warnings, self.warnings = self.warnings, []
        for warning in warnings:
            db.log_warning(warning)
        if not pending:
            return False
        import omni.ui as ui
        for region, (emission, base) in pending.items():
            providers = self.providers.get(region)
            if providers is None:
                providers = self.providers[region] = (ui.DynamicTextureProvider(TEXTURE_PREFIX + region),
                                                      ui.DynamicTextureProvider(TEXTURE_PREFIX + region + "_base"))
                self._use_dynamic_texture(db, region)
            for provider, data in zip(providers, (emission, base)):
                provider.set_data_array(data, [data.shape[1], data.shape[0]])
        return True

    @staticmethod
    def _use_dynamic_texture(db, region: str) -> None:
        """Point the region's emission, opacity and base color textures at its dynamic textures (once).
        The panels are the meshes with face:region = <id>, wherever the robot sits on the stage."""
        import carb
        import omni.usd
        from pxr import Sdf, Usd, UsdShade

        stage = omni.usd.get_context().get_stage()
        found = False
        for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
            attr = prim.GetAttribute(REGION_ATTR)
            if not attr or attr.Get() != region:
                continue
            material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            shader = stage.GetPrimAtPath(material.GetPath().AppendChild("Shader")) if material else None
            if shader and shader.IsValid():
                url = Sdf.AssetPath(f"dynamic://{TEXTURE_PREFIX}{region}")
                shader.GetAttribute("inputs:emissive_color_texture").Set(url)
                shader.GetAttribute("inputs:opacity_texture").Set(url)
                shader.GetAttribute("inputs:diffuse_texture").Set(Sdf.AssetPath(f"dynamic://{TEXTURE_PREFIX}{region}_base"))
                found = True
                carb.log_info(f"face display: {prim.GetName()} shows {url.path}")
        if not found:
            db.log_warning(f"no panel with {REGION_ATTR} = {region} on the stage (face_display.usda)")


class OgnROS2SubscribeFaceExpression:
    """OmniGraph node that shows the newest face expression images on the face's LED panels."""

    @staticmethod
    def internal_state() -> OgnROS2SubscribeFaceExpressionInternalState:
        """Create the per-node internal state."""
        return OgnROS2SubscribeFaceExpressionInternalState()

    @staticmethod
    def compute(db) -> bool:
        """(Re)create the subscription on input changes, then upload the textures of new images."""
        import omni.graph.core as og

        state = db.per_instance_state
        try:
            domain_id = _resolve_domain_id(db)
        except Exception as exc:
            db.log_error(f"Failed to resolve ROS 2 domain ID: {exc}")
            return False
        ros_key = (db.inputs.nodeName, db.inputs.nodeNamespace, db.inputs.topicName, domain_id, db.inputs.qosProfile)

        # a failure is latched until an input changes, so errors are not repeated every tick
        if ros_key == state.failed_key:
            return False

        if ros_key != state.ros_key:
            state.shutdown_ros()
            try:
                state.start_ros(db.inputs.nodeName, db.inputs.nodeNamespace, db.inputs.topicName, domain_id,
                                _qos_from_json(db.inputs.qosProfile))
            except Exception as exc:
                db.log_error(f"Failed to create ROS 2 subscriber: {exc}")
                state.shutdown_ros()
                state.failed_key = ros_key
                return False
            state.ros_key = ros_key
        state.failed_key = None

        updated = state.upload(db)
        db.outputs.execOut = og.ExecutionAttributeState.ENABLED if updated else og.ExecutionAttributeState.DISABLED
        return True

    @staticmethod
    def release_instance(node, graph_instance_id: int) -> None:
        """Shut down the ROS 2 node when the node instance is removed."""
        from isaacsim.nicol.nodes.ogn.OgnROS2SubscribeFaceExpressionDatabase import (
            OgnROS2SubscribeFaceExpressionDatabase,
        )

        try:
            state = OgnROS2SubscribeFaceExpressionDatabase.get_internal_state(node, graph_instance_id)
        except Exception:
            state = None
        if state is not None:
            state.shutdown_ros()
            state.providers.clear()


if __name__ == "__main__":
    import sys
    import cv2
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    bgra = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
    rgba = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA)
    emission, base = led_textures(rgba)
    cv2.imwrite(sys.argv[2], cv2.cvtColor(emission, cv2.COLOR_RGBA2BGRA))
    root, ext = sys.argv[2].rsplit(".", 1)
    cv2.imwrite(f"{root}_base.{ext}", cv2.cvtColor(base, cv2.COLOR_RGBA2BGRA))
