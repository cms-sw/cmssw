#include "SimCalorimetry/HGCalSimProducers/interface/HGCDigitizerPluginFactory.h"
#include "HGCEEDigitizer.h"
#include "HGCHEfrontDigitizer.h"
#include "HGCHEbackDigitizer.h"
#include "HFNoseDigitizer.h"

EDM_REGISTER_PLUGINFACTORY(HGCDigitizerPluginFactory, "HGCDigitizerPluginFactory");
DEFINE_EDM_PLUGIN(HGCDigitizerPluginFactory, HGCEEDigitizer, "HGCEEDigitizer");
DEFINE_EDM_PLUGIN(HGCDigitizerPluginFactory, HGCHEfrontDigitizer, "HGCHEfrontDigitizer");
DEFINE_EDM_PLUGIN(HGCDigitizerPluginFactory, HGCHEbackDigitizer, "HGCHEbackDigitizer");
DEFINE_EDM_PLUGIN(HGCDigitizerPluginFactory, HFNoseDigitizer, "HFNoseDigitizer");
