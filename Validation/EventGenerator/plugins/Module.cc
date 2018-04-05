#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Validation/EventGenerator/interface/BasicHepMCValidation.h"
DEFINE_FWK_MODULE (BasicHepMCValidation);

#include "Validation/EventGenerator/interface/BasicHepMCHeavyIonValidation.h"
DEFINE_FWK_MODULE (BasicHepMCHeavyIonValidation);

#include "Validation/EventGenerator/interface/BasicGenParticleValidation.h"
DEFINE_FWK_MODULE(BasicGenParticleValidation);

#include "Validation/EventGenerator/interface/DuplicationChecker.h"
DEFINE_FWK_MODULE (DuplicationChecker);

#include "MBUEandQCDValidation.h"
DEFINE_FWK_MODULE (MBUEandQCDValidation);

#include "Validation/EventGenerator/interface/DrellYanValidation.h"
DEFINE_FWK_MODULE (DrellYanValidation);

#include "Validation/EventGenerator/interface/WValidation.h"
DEFINE_FWK_MODULE (WValidation);

#include "Validation/EventGenerator/interface/TauValidation.h"
DEFINE_FWK_MODULE (TauValidation);

#include "Validation/EventGenerator/interface/TTbar_GenJetAnalyzer.h"
DEFINE_FWK_MODULE(TTbar_GenJetAnalyzer);

#include "Validation/EventGenerator/interface/TTbar_GenLepAnalyzer.h"
DEFINE_FWK_MODULE(TTbar_GenLepAnalyzer);

#include "Validation/EventGenerator/interface/TTbar_Kinematics.h"
DEFINE_FWK_MODULE(TTbar_Kinematics);

#include "Validation/EventGenerator/interface/TTbarSpinCorrHepMCAnalyzer.h"
DEFINE_FWK_MODULE(TTbarSpinCorrHepMCAnalyzer);

#include "Validation/EventGenerator/interface/HiggsValidation.h"
DEFINE_FWK_MODULE(HiggsValidation);

#include "Validation/EventGenerator/interface/BPhysicsValidation.h"
DEFINE_FWK_MODULE(BPhysicsValidation);

#include "Validation/EventGenerator/interface/BPhysicsSpectrum.h"
DEFINE_FWK_MODULE(BPhysicsSpectrum);
