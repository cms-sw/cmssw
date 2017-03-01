#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"

#include "SimG4Core/Physics/interface/PhysicsListFactory.h"
#include "SimG4Core/CustomPhysics/interface/CustomPhysics.h"

#include "SimG4Core/CustomPhysics/interface/RHStopDump.h"
#include "SimG4Core/CustomPhysics/interface/RHStopTracer.h"

DEFINE_PHYSICSLIST(CustomPhysics);
DEFINE_FWK_MODULE(RHStopDump) ;
DEFINE_SIMWATCHER(RHStopTracer);

