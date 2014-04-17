#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "Validation/Geometry/interface/MaterialBudget.h"
#include "Validation/Geometry/interface/MaterialBudgetAction.h"
#include "Validation/Geometry/interface/MaterialBudgetHcal.h"
#include "Validation/Geometry/interface/MaterialBudgetForward.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

DEFINE_SIMWATCHER (MaterialBudget);
DEFINE_SIMWATCHER (MaterialBudgetAction);
DEFINE_SIMWATCHER (MaterialBudgetHcal);
DEFINE_SIMWATCHER (MaterialBudgetForward);

