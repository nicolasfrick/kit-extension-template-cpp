// OgnIsaacComputeTransformTreeHierarchical.cpp
//
// Computes a robot_state_publisher-style TF tree: each target's parent is either
// an explicit override or the nearest USD ancestor that is itself a target,
// falling back to `parentPrim` (or World) at the root. Deliberately independent
// of PhysX/UsdPhysicsArticulationRootAPI/joint-graph discovery -- this walks
// plain USD prim hierarchy only, so it does not require PhysicsJoint prims to
// exist for a parent/child relationship to be reported correctly.

#include <OgnIsaacComputeTransformTreeHierarchicalDatabase.h>

#include <omni/fabric/FabricUSD.h>

#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/quatd.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/vec4d.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/xformCache.h>
#include <pxr/usd/usdUtils/stageCache.h>

#include <string>
#include <unordered_set>
#include <vector>

namespace isaacsim
{
namespace nicol
{
namespace nodes
{

class OgnIsaacComputeTransformTreeHierarchical
{
public:
    static bool compute(OgnIsaacComputeTransformTreeHierarchicalDatabase& db)
    {
        const auto& targetPrimsIn = db.inputs.targetPrims();
        if (targetPrimsIn.empty())
        {
            db.logError("Please specify at least one valid target prim");
            return false;
        }

        // --- Resolve the stage. ---
        long stageId = db.abi_context().iContext->getStageId(db.abi_context());
        pxr::UsdStageWeakPtr stage =
            pxr::UsdUtilsStageCache::Get().Find(pxr::UsdStageCache::Id::FromLongInt(stageId));
        if (!stage)
        {
            db.logError("Could not find USD stage %ld", stageId);
            return false;
        }

        // --- Resolve target paths, preserving order, skipping invalid entries. ---
        std::vector<pxr::SdfPath> targetPaths;
        targetPaths.reserve(targetPrimsIn.size());
        std::unordered_set<pxr::SdfPath, pxr::SdfPath::Hash> targetPathSet;

        for (size_t i = 0; i < targetPrimsIn.size(); i++)
        {
            pxr::SdfPath path = omni::fabric::toSdfPath(targetPrimsIn[i]);
            if (path.IsEmpty() || !stage->GetPrimAtPath(path))
            {
                CARB_LOG_WARN(
                    "IsaacComputeTransformTreeHierarchical: skipping invalid/missing target at index %zu", i);
                continue;
            }
            targetPaths.push_back(path);
            targetPathSet.insert(path);
        }

        if (targetPaths.empty())
        {
            db.logError("No valid prims found after resolving target prims");
            return false;
        }

        // --- Per-target override strings (token[]; may be shorter than targetPaths). ---
        const auto& overridesIn = db.inputs.parentOverrides();

        // --- Fallback root parent (parentPrim input), defaulting to World. ---
        pxr::SdfPath fallbackParentPath = pxr::SdfPath::AbsoluteRootPath();
        const auto& parentPrimIn = db.inputs.parentPrim();
        if (!parentPrimIn.empty())
        {
            pxr::SdfPath p = omni::fabric::toSdfPath(parentPrimIn[0]);
            if (!p.IsEmpty() && stage->GetPrimAtPath(p))
            {
                fallbackParentPath = p;
            }
        }

        pxr::UsdGeomXformCache xformCache;

        auto worldTransform = [&](const pxr::SdfPath& path) -> pxr::GfMatrix4d
        {
            if (path == pxr::SdfPath::AbsoluteRootPath())
            {
                return pxr::GfMatrix4d(1.0); // identity: World
            }
            return xformCache.GetLocalToWorldTransform(stage->GetPrimAtPath(path));
        };

        // Mirrors isaacsim::core::includes::getName(): a non-empty `isaac:nameOverride` wins over the prim name.
        static const pxr::TfToken kNameOverride("isaac:nameOverride");
        auto frameName = [&](const pxr::SdfPath& path) -> std::string
        {
            if (path == pxr::SdfPath::AbsoluteRootPath())
            {
                return "world";
            }
            if (pxr::UsdPrim prim = stage->GetPrimAtPath(path))
            {
                std::string nameOverride;
                pxr::UsdAttribute attr = prim.GetAttribute(kNameOverride);
                if (attr && attr.Get(&nameOverride) && !nameOverride.empty())
                {
                    return nameOverride;
                }
            }
            return path.GetName();
        };

        std::vector<NameToken> parentFramesOut;
        std::vector<NameToken> childFramesOut;
        std::vector<pxr::GfVec3d> translationsOut;
        std::vector<pxr::GfVec4d> orientationsOut; // (x, y, z, w)
        parentFramesOut.reserve(targetPaths.size());
        childFramesOut.reserve(targetPaths.size());
        translationsOut.reserve(targetPaths.size());
        orientationsOut.reserve(targetPaths.size());

        for (size_t i = 0; i < targetPaths.size(); i++)
        {
            const pxr::SdfPath& childPath = targetPaths[i];
            pxr::SdfPath parentPath;
            bool resolved = false;

            // Step 1: explicit override at this index, if present and non-empty.
            if (i < overridesIn.size())
            {
                const std::string overrideStr = db.tokenToString(overridesIn[i]);
                if (!overrideStr.empty())
                {
                    pxr::SdfPath p(overrideStr);
                    if (stage->GetPrimAtPath(p))
                    {
                        parentPath = p;
                        resolved = true;
                    }
                    else
                    {
                        CARB_LOG_WARN(
                            "IsaacComputeTransformTreeHierarchical: parentOverrides[%zu] '%s' is not a valid "
                            "prim, falling back to hierarchy walk",
                            i, overrideStr.c_str());
                    }
                }
            }

            // Step 2: nearest USD ancestor that is itself a target.
            if (!resolved)
            {
                pxr::UsdPrim ancestor = stage->GetPrimAtPath(childPath).GetParent();
                while (ancestor && ancestor.GetPath() != pxr::SdfPath::AbsoluteRootPath())
                {
                    if (targetPathSet.count(ancestor.GetPath()))
                    {
                        parentPath = ancestor.GetPath();
                        resolved = true;
                        break;
                    }
                    ancestor = ancestor.GetParent();
                }
            }

            // Step 3: fallback to parentPrim input (or World).
            if (!resolved)
            {
                parentPath = fallbackParentPath;
            }

            // Relative transform: child pose expressed in parent's frame.
            const pxr::GfMatrix4d childWorld = worldTransform(childPath);
            const pxr::GfMatrix4d parentWorld = worldTransform(parentPath);
            const pxr::GfMatrix4d relative = childWorld * parentWorld.GetInverse();

            const pxr::GfVec3d translation = relative.ExtractTranslation();
            const pxr::GfQuatd quat = relative.RemoveScaleShear().ExtractRotationQuat();
            const pxr::GfVec3d imag = quat.GetImaginary();

            parentFramesOut.push_back(db.stringToToken(frameName(parentPath).c_str()));
            childFramesOut.push_back(db.stringToToken(frameName(childPath).c_str()));
            translationsOut.push_back(translation);
            orientationsOut.push_back(pxr::GfVec4d(imag[0], imag[1], imag[2], quat.GetReal()));
        }

        // --- Write outputs. ---
        db.outputs.parentFrames().resize(parentFramesOut.size());
        db.outputs.childFrames().resize(childFramesOut.size());
        db.outputs.translations().resize(translationsOut.size());
        db.outputs.orientations().resize(orientationsOut.size());

        for (size_t i = 0; i < parentFramesOut.size(); i++)
        {
            db.outputs.parentFrames()[i] = parentFramesOut[i];
            db.outputs.childFrames()[i] = childFramesOut[i];
            db.outputs.translations()[i] = translationsOut[i];
            db.outputs.orientations()[i] = orientationsOut[i];
        }

        db.outputs.execOut() = kExecutionAttributeStateEnabled;
        return true;
    }
};

REGISTER_OGN_NODE()

} // namespace nodes
} // namespace nicol
} // namespace isaacsim
