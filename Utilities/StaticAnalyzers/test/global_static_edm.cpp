
#include <string>

 namespace edmplugin {
   
 template<class R>
 class PluginFactory
 { };
}

namespace edm {
  
   typedef edmplugin::PluginFactory<int> InputSourcePluginFactory;	
}

// is ok, because const-qualified
const static int g_staticConst = 23;

// results in a warning by GlobalStaticChecker
static int g_static;

// results in a warning by GlobalStaticChecker
static edm::InputSourcePluginFactory g_static_edm_namespace;



int main()
{
    g_static = 23;

    return 0;
}
