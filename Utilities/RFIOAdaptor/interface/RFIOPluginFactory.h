#ifndef RFIO_ADAPTOR_RFIO_PLUGIN_FACTORY_H
# define RFIO_ADAPTOR_RFIO_PLUGIN_FACTORY_H

#include "FWCore/PluginManager/interface/PluginFactory.h"

struct RFIODummyFile {};
typedef edmplugin::PluginFactory<RFIODummyFile*(void)> RFIOPluginFactory;

#endif // RFIO_ADAPTOR_RFIO_PLUGIN_FACTORY_H
