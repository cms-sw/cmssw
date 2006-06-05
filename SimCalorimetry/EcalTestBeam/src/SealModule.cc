/*  
 *  $Id:$
 */

#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "SimCalorimetry/EcalTestBeam/interface/EcalTBDigiProducer.h"

DEFINE_SEAL_MODULE () ;

using edm::EDProducer;
DEFINE_ANOTHER_FWK_MODULE (EcalTBDigiProducer) ;


