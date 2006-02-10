/*  
 *  \author  P. Govoni Univ Milano Bicocca - INFN Milano 
 */

#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "SimCalorimetry/EcalSimProducers/interface/EcalDigiProducer.h"
#include "SimCalorimetry/EcalSimProducers/interface/EcalDigiTester.h"

DEFINE_SEAL_MODULE () ;
DEFINE_ANOTHER_FWK_MODULE (EcalDigiProducer) ;
DEFINE_ANOTHER_FWK_MODULE (EcalDigiTester) ;

