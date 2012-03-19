#include "Utilities/RFIOAdaptor/interface/RFIOPluginFactory.h"
#include <dlfcn.h>
namespace {
// dlopen libdpm on startup, so that symbols get resolved even if we don't
// explicitly link against it. This way we do not need to have
// libdpm available at buildtime.
struct Dummy
{
  Dummy()
  {
#ifdef __APPLE__
  dlopen("libdpm.dylib", RTLD_NOW|RTLD_GLOBAL);
#else
  dlopen("libdpm.so", RTLD_NOW|RTLD_GLOBAL);
#endif
  }
};
static Dummy foo;
}
DEFINE_EDM_PLUGIN (RFIOPluginFactory,RFIODummyFile,"dpm");
