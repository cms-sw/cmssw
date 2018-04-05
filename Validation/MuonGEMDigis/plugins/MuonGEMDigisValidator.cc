#include "Validation/MuonGEMDigis/interface/GEMStripDigiValidation.h"
#include "Validation/MuonGEMDigis/interface/GEMPadDigiValidation.h"
#include "Validation/MuonGEMDigis/interface/GEMCoPadDigiValidation.h"
#include "Validation/MuonGEMDigis/interface/GEMDigiTrackMatch.h"
#include "Validation/MuonGEMDigis/interface/GEMCheckGeometry.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE (GEMStripDigiValidation) ;
DEFINE_FWK_MODULE (GEMPadDigiValidation) ;
DEFINE_FWK_MODULE (GEMCoPadDigiValidation) ;
DEFINE_FWK_MODULE (GEMDigiTrackMatch) ;
DEFINE_FWK_MODULE (GEMCheckGeometry) ;

