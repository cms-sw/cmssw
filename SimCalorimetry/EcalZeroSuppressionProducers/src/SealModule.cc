/*  
 *  \author  F. Cossutti - INFN Trieste
 */

#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "SimCalorimetry/EcalZeroSuppressionProducers/interface/EcalZeroSuppressionProducer.h"

DEFINE_SEAL_MODULE () ;
DEFINE_ANOTHER_FWK_MODULE (EcalZeroSuppressionProducer) ;

