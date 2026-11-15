```{csv-table}
**Extension**: {{ extension_version }},**Documentation Generated**: {sub-ref}`today`,{ref}`changelog_omni_example_cpp_omnigraph_node`
```

(ext_omni_example_cpp_omnigraph_node)=

# Overview

An [example C++ extension](http://omniverse-docs.s3-website-us-east-1.amazonaws.com/kit-extension-template-cpp/106.0.1/index.html) that can be used as a reference/template for creating new extensions.

Demonstrates how to create a C++ node for omni.graph

## NICOL nodes

| node type `isaacsim.nicol.nodes.` | language | what |
|---|---|---|
| `IsaacComputeTransformTreeHierarchical` | C++ | robot_state_publisher-style TF tree from the USD hierarchy (parentOverrides, nearest target ancestor, parentPrim) |
| `ROS2PublishRobotDescription` | Python | the URDF (xacro expanded) latched on `robot_description` |
| `ROS2SubscribeFaceExpression` | Python | nicol_face_expression's images as dynamic textures on the face's LED panels |
| `ROS2PublishFingertipForces` | Python | RH8D fingertip force sensors: PhysX contact report of the tip bodies (they need `PhysxContactReportAPI`), net force per tip in the tip's frame x `forceScale` (N -> mN), published as `ros2_sensor_pkg/AllSensors` at `publishRate` in simulation time; contact pairs kept from found to lost (sleeping bodies keep their force); `includeFriction` adds the shear impulses |
| `ROS2PublishPalmLaser` | Python | RH8D palm TOF: one PhysX `raycast_all` from `sensorPrim` along its +x, the sensor's own body skipped (`ignoreHand`: its parent's bodies), clipped to [`minRange`, `maxRange`], published as `nicol_msgs/PalmLaser` x `rangeScale` (m -> mm) |

The Python nodes create their ROS 2 node on an own rclpy context (domain from the connected ROS2 Context node).
The message packages (`ros2_sensor_pkg`, `nicol_msgs`) must be importable in the Isaac Sim process (the NICOL
workspace sourced). Usage and measurements: nicol_ros `nicol_isaac/README.md` ("Hand sensors", "Face expressions").
Build: `./build.sh -r`, then copy `_build/linux-x86_64/release/exts/isaacsim.nicol.nodes` into
`nicol_isaac/extensions/` and `colcon build --packages-select nicol_isaac`.
