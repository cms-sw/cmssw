#ifndef __APPLE__
#include "Utilities/RFIOAdaptor/interface/RFIOPluginFactory.h"
DEFINE_EDM_PLUGIN (RFIOPluginFactory,RFIODummyFile,"dpm");
#else // ! __APPLE__
int dummyDPMPlugin() { return 0; }
#endif // ! __APPLE__
