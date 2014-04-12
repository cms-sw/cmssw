#ifndef STORAGE_FACTORY_STORAGE_MAKER_FACTORY_H
# define STORAGE_FACTORY_STORAGE_MAKER_FACTORY_H

# include "FWCore/PluginManager/interface/PluginFactory.h"

class StorageMaker;
typedef edmplugin::PluginFactory<StorageMaker *(void)> StorageMakerFactory;

#endif // STORAGE_FACTORY_STORAGE_MAKER_FACTORY_H
