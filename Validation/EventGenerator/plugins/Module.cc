#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Validation/EventGenerator/interface/BasicHepMCValidation.h"
DEFINE_FWK_MODULE (BasicHepMCValidation);

#include "Validation/EventGenerator/interface/BasicGenParticleValidation.h"
DEFINE_FWK_MODULE(BasicGenParticleValidation);

#include "Validation/EventGenerator/interface/DuplicationChecker.h"
DEFINE_FWK_MODULE (DuplicationChecker);

#include "Validation/EventGenerator/interface/MBUEandQCDValidation.h"
DEFINE_FWK_MODULE (MBUEandQCDValidation);

#include "Validation/EventGenerator/interface/DrellYanValidation.h"
DEFINE_FWK_MODULE (DrellYanValidation);

#include "Validation/EventGenerator/interface/WValidation.h"
DEFINE_FWK_MODULE (WValidation);

#include "Validation/EventGenerator/interface/TauValidation.h"
DEFINE_FWK_MODULE (TauValidation);
