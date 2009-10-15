/*  
 *  $Id: SealModule.cc,v 1.4 2009/09/11 17:37:39 heltsley Exp $
 */

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "SimCalorimetry/EcalTestBeam/interface/EcalTBDigiProducer.h"

DEFINE_SEAL_MODULE () ;

using edm::EDProducer;
DEFINE_ANOTHER_FWK_MODULE (EcalTBDigiProducer) ;


