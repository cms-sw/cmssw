/*  
 *  \author  F. Cossutti - INFN Trieste
 */

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"

#include "SimCalorimetry/EcalSelectiveReadoutProducers/interface/EcalSelectiveReadoutProducer.h"
#include "SimCalorimetry/EcalSelectiveReadoutProducers/interface/EcalSRCondTools.h"
#include "CondFormats/EcalObjects/interface/EcalSRSettings.h"

DEFINE_FWK_MODULE(EcalSelectiveReadoutProducer);
DEFINE_FWK_MODULE(EcalSRCondTools);
