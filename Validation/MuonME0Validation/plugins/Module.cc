// -*- C++ -*-
//
// Package:    MuonME0Hits
// Class:      MuonME0Hits
//
/**\class Module.cc

 Description: Declaration of the ME0 plugins for the local validation

 Implementation:
     [Notes on implementation]
*/
//
// Original Author: Claudio Caputo, INFN Bari
//         Created:  14 Jan 2016 17:00:00 GMT
// $Id$
//
//
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "Validation/MuonME0Validation/interface/ME0DigisValidation.h"
#include "Validation/MuonME0Validation/interface/ME0HitsValidation.h"
#include "Validation/MuonME0Validation/interface/ME0RecHitsValidation.h"
#include "Validation/MuonME0Validation/interface/ME0SegmentsValidation.h"
DEFINE_FWK_MODULE(ME0HitsValidation);
DEFINE_FWK_MODULE(ME0DigisValidation);
DEFINE_FWK_MODULE(ME0RecHitsValidation);
DEFINE_FWK_MODULE(ME0SegmentsValidation);
