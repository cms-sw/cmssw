#include "SimG4CMS/Forward/interface/TotemTestGem.h"
#include "SimG4CMS/Forward/interface/CastorTestAnalysis.h"
#include "SimG4CMS/Forward/interface/ZdcTestAnalysis.h"
#include "SimG4CMS/Forward/interface/DoCastorAnalysis.h"
#include "SimG4CMS/Forward/interface/BscTest.h"
#include "SimG4CMS/Forward/interface/SimG4FluxProducer.h"

#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

DEFINE_SIMWATCHER(CastorTestAnalysis);
DEFINE_SIMWATCHER(ZdcTestAnalysis);
DEFINE_SIMWATCHER(DoCastorAnalysis);
DEFINE_SIMWATCHER(TotemTestGem);
DEFINE_SIMWATCHER(BscTest);
DEFINE_SIMWATCHER(SimG4FluxProducer);
