/*  
 *  $Id: SealModule.cc,v 1.3 2009/06/15 19:47:00 heltsley Exp $
 */

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "SimCalorimetry/EcalTestBeam/interface/EcalTBDigiProducer.h"
#include "SimCalorimetry/EcalTestBeam/interface/EETBDigiProducer.h"

DEFINE_SEAL_MODULE () ;

using edm::EDProducer;
DEFINE_ANOTHER_FWK_MODULE (EcalTBDigiProducer) ;


