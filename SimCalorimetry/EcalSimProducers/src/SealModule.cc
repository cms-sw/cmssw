/*  
 *  \author  P. Govoni Univ Milano Bicocca - INFN Milano 
 */

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "SimCalorimetry/EcalSimProducers/interface/EcalDigiProducer.h"

DEFINE_SEAL_MODULE ();
DEFINE_ANOTHER_FWK_MODULE (EcalDigiProducer);

