#include "TauAnalysis/MCEmbeddingTools/plugins/CaloRecHitMixer.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"

#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HORecHit.h"
#include "DataFormats/HcalRecHit/interface/ZDCRecHit.h"
#include "DataFormats/HcalRecHit/interface/CastorRecHit.h"


typedef CaloRecHitMixer< EcalRecHit > EcalRHMixer;
typedef CaloRecHitMixer< HBHERecHit > HBHERHMixer;
typedef CaloRecHitMixer< HFRecHit >   HFRHMixer;
typedef CaloRecHitMixer< HORecHit > HORHMixer;
#warning "ZDCRHMixer still needs to be done" 
//typedef CaloRecHitMixer< ZDCRecHit > ZDCRHMixer;
typedef CaloRecHitMixer< CastorRecHit > CastorRHMixer;


DEFINE_FWK_MODULE(EcalRHMixer);
DEFINE_FWK_MODULE(HBHERHMixer);
DEFINE_FWK_MODULE(HFRHMixer);
DEFINE_FWK_MODULE(HORHMixer);
//DEFINE_FWK_MODULE(ZDCRHMixer);
DEFINE_FWK_MODULE(CastorRHMixer);


