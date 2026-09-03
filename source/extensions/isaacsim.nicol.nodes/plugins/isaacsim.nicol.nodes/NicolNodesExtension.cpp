#define CARB_EXPORTS

#include <carb/PluginUtils.h>

#include <omni/ext/IExt.h>
#include <omni/graph/core/IGraphRegistry.h>
#include <omni/graph/core/ogn/Database.h>
#include <omni/graph/core/ogn/Registration.h>

// Standard plugin definitions required by Carbonite.
const struct carb::PluginImplDesc pluginImplDesc = { "isaacsim.nicol.nodes.plugin",
                                                     "NICOL robot OmniGraph nodes.", "NICOL",
                                                     carb::PluginHotReload::eEnabled, "dev" };

// These interface dependencies are required by all OmniGraph node types
CARB_PLUGIN_IMPL_DEPS(omni::graph::core::IGraphRegistry,
                      omni::fabric::IPath,
                      omni::fabric::IToken)

// This macro sets up the information required to register your node type definitions with OmniGraph
DECLARE_OGN_NODES()

namespace isaacsim
{
namespace nicol
{
namespace nodes
{

class NicolNodesExtension : public omni::ext::IExt
{
public:
    void onStartup(const char* extId) override
    {
        printf("isaacsim.nicol.nodes starting up (ext_id: %s).\n", extId);
        // This macro walks the list of pending node type definitions and registers them with OmniGraph
        INITIALIZE_OGN_NODES()
    }

    void onShutdown() override
    {
        printf("isaacsim.nicol.nodes shutting down.\n");
        // This macro walks the list of registered node type definitions and deregisters all of them. This is required
        // for hot reload to work.
        RELEASE_OGN_NODES()
    }

private:
};

}
}
}

CARB_PLUGIN_IMPL(pluginImplDesc, isaacsim::nicol::nodes::NicolNodesExtension)

void fillInterface(isaacsim::nicol::nodes::NicolNodesExtension& iface)
{
}
