/*  
 *  \author  F. Cossutti - INFN Trieste
 */

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"

#include "SimCalorimetry/EcalSelectiveReadoutProducers/interface/EcalSelectiveReadoutProducer.h"
#include "SimCalorimetry/EcalSelectiveReadoutProducers/interface/EcalSRCondTools.h"
#include "SimCalorimetry/EcalSelectiveReadoutProducers/interface/EcalSRSimpleESSource.h"

DEFINE_FWK_MODULE(EcalSelectiveReadoutProducer);

DEFINE_FWK_MODULE(EcalSRCondTools);

DEFINE_FWK_EVENTSETUP_SOURCE(EcalSRSimpleESSource);

