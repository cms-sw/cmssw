#include "SimG4Core/PrintGeomInfo/interface/PrintGeomInfoAction.h"
#include "SimG4Core/PrintGeomInfo/interface/PrintGeomSummary.h"
#include "SimG4Core/PrintGeomInfo/interface/PrintMaterialBudgetInfo.h"
#include "SimG4Core/PrintGeomInfo/interface/PrintSensitive.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

DEFINE_SIMWATCHER(PrintGeomInfoAction);
DEFINE_SIMWATCHER(PrintGeomSummary);
DEFINE_SIMWATCHER(PrintMaterialBudgetInfo);
DEFINE_SIMWATCHER(PrintSensitive);
