/*  
 *  $Id: SealModule.cc,v 1.1 2006/06/05 14:00:52 fabiocos Exp $
 */

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "SimCalorimetry/EcalTestBeam/interface/EcalTBDigiProducer.h"

DEFINE_SEAL_MODULE () ;

using edm::EDProducer;
DEFINE_ANOTHER_FWK_MODULE (EcalTBDigiProducer) ;


