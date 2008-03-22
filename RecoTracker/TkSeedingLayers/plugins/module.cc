#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

#include "PixelLayerPairsESProducer.h"
#include "MixedLayerPairsESProducer.h"
#include "PixelLayerTripletsESProducer.h"
#include "PixelLessLayerPairsESProducer.h"
#include "MixedLayerTripletsESProducer.h"
#include "TobTecLayerPairsESProducer.h"


DEFINE_FWK_EVENTSETUP_MODULE(PixelLayerPairsESProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(MixedLayerPairsESProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(MixedLayerTripletsESProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(PixelLayerTripletsESProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(PixelLessLayerPairsESProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(TobTecLayerPairsESProducer);

