/*  
 *  $Id: SealModule.cc,v 1.2 2007/04/08 03:18:55 wmtan Exp $
 */

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "SimCalorimetry/EcalTestBeam/interface/EcalTBDigiProducer.h"
#include "SimCalorimetry/EcalTestBeam/interface/EETBDigiProducer.h"

DEFINE_SEAL_MODULE () ;

using edm::EDProducer;
DEFINE_ANOTHER_FWK_MODULE (EcalTBDigiProducer) ;
DEFINE_ANOTHER_FWK_MODULE (EETBDigiProducer) ;


