#include "SimG4Core/PrintGeomInfo/interface/PrintGeomInfoAction.h"
#include "SimG4Core/PrintGeomInfo/interface/PrintMaterialBudgetInfo.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "PluginManager/ModuleDef.h"
   
DEFINE_SEAL_MODULE ();
DEFINE_SIMWATCHER(PrintGeomInfoAction);
DEFINE_SIMWATCHER(PrintMaterialBudgetInfo);
