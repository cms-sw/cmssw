#include "TauAnalysis/MCEmbeddingTools/plugins/CaloRecHitMixer.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HORecHit.h"
#include "DataFormats/HcalRecHit/interface/ZDCRecHit.h"
#include "DataFormats/HcalRecHit/interface/CastorRecHit.h"

typedef CaloRecHitMixer<EcalRecHit> EcalRecHitMixer;
typedef CaloRecHitMixer<HBHERecHit> HBHERecHitMixer;
typedef CaloRecHitMixer<HFRecHit> HFRecHitMixer;
typedef CaloRecHitMixer<HORecHit> HORecHitMixer;
//typedef CaloRecHitMixer<ZDCRecHit> ZDCRecHitMixer;
typedef CaloRecHitMixer<CastorRecHit> CastorRecHitMixer;

//-------------------------------------------------------------------------------
// define 'buildRecHit' functions used for different types of recHits
//-------------------------------------------------------------------------------

template <typename T>
T CaloRecHitMixer<T>::buildRecHit(const CaloRecHitMixer_mixedRecHitInfoType<T>& recHitInfo)
{
  assert(0); // CV: make sure general function never gets called;
             //     always use template specializations
}

// template specialization for ECAL recHits
//
// NOTE: rh1 should refer to simulated tau decay products,
//       rh2 to Zmumu event
//
template <>
EcalRecHit CaloRecHitMixer<EcalRecHit>::buildRecHit(const CaloRecHitMixer_mixedRecHitInfoType<EcalRecHit>& recHitInfo)
{
  // CV: take status flags and timing information from simulated tau decay products
  //    (suggested by Florian Beaudette)
  const EcalRecHit* recHit_ecal = NULL;
  if      ( recHitInfo.isRecHit1_ ) recHit_ecal = recHitInfo.recHit1_;
  else if ( recHitInfo.isRecHit2_ ) recHit_ecal = recHitInfo.recHit2_;

  assert(recHit_ecal);
  assert(recHitInfo.isRecHitSum_);

  EcalRecHit mergedRecHit(*recHit_ecal);
  mergedRecHit.setEnergy(recHitInfo.energySum_);

  /* TODO Does not make sense
  uint32_t flagBits = 0;
  for ( int flag = 0; flag < 32; ++flag ) {
    if ( recHit_ecal->checkFlag(flag) ) flagBits += (0x1 << flag);
  }
  EcalRecHit mergedRecHit(recHit_ecal->detid(), recHitInfo.energySum_, recHit_ecal->time(), recHit_ecal->flags(), flagBits);
  mergedRecHit.setAux(recHit_ecal->aux());
  */

  return mergedRecHit;
}

// template specialization for different types of HCAL recHits
//
// NOTE: rh1 should refer to simulated tau decay products,
//       rh2 to Zmumu event
//
namespace
{
  template <typename T>
  T buildRecHit_HCAL(const CaloRecHitMixer_mixedRecHitInfoType<T>& recHitInfo) 
  {
    // CV: take status flags and timing information from simulated tau decay products
    //    (suggested by Florian Beaudette)
    const CaloRecHit* recHit = 0;
    if      ( recHitInfo.isRecHit1_ ) recHit = recHitInfo.recHit1_;
    else if ( recHitInfo.isRecHit2_ ) recHit = recHitInfo.recHit2_;
    const T* recHit_hcal = static_cast<const T*>(recHit);
    assert(recHit_hcal);
    assert(recHitInfo.isRecHitSum_);
    T mergedRecHit(recHit_hcal->detid(), recHitInfo.energySum_, recHit_hcal->time());
    mergedRecHit.setFlags(recHit_hcal->flags());
    mergedRecHit.setAux(recHit_hcal->aux());
    return mergedRecHit;
  }
}

template <>
HBHERecHit CaloRecHitMixer<HBHERecHit>::buildRecHit(const CaloRecHitMixer_mixedRecHitInfoType<HBHERecHit>& recHitInfo)
{
  return buildRecHit_HCAL<HBHERecHit>(recHitInfo);
}

template <>
HORecHit CaloRecHitMixer<HORecHit>::buildRecHit(const CaloRecHitMixer_mixedRecHitInfoType<HORecHit>& recHitInfo)
{
  return buildRecHit_HCAL<HORecHit>(recHitInfo);
}

template <>
HFRecHit CaloRecHitMixer<HFRecHit>::buildRecHit(const CaloRecHitMixer_mixedRecHitInfoType<HFRecHit>& recHitInfo)
{
  return buildRecHit_HCAL<HFRecHit>(recHitInfo);
}

// template specialization for CASTOR recHits
//
// NOTE: rh1 should refer to simulated tau decay products,
//       rh2 to Zmumu event
//
template <>
CastorRecHit CaloRecHitMixer<CastorRecHit>::buildRecHit(const CaloRecHitMixer_mixedRecHitInfoType<CastorRecHit>& recHitInfo)
{
  // CV: take status flags and timing information from simulated tau decay products
  //    (suggested by Florian Beaudette)
  const CaloRecHit* recHit = 0;
  if      ( recHitInfo.isRecHit1_ ) recHit = recHitInfo.recHit1_;
  else if ( recHitInfo.isRecHit2_ ) recHit = recHitInfo.recHit2_;
  const CastorRecHit* recHit_castor = static_cast<const CastorRecHit*>(recHit);
  assert(recHit_castor);
  assert(recHitInfo.isRecHitSum_);
  CastorRecHit mergedRecHit(recHit_castor->detid(), recHitInfo.energySum_, recHit_castor->time());
  mergedRecHit.setFlags(recHit_castor->flags());
  mergedRecHit.setAux(recHit_castor->aux());
  return mergedRecHit;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(EcalRecHitMixer);
DEFINE_FWK_MODULE(HBHERecHitMixer);
DEFINE_FWK_MODULE(HFRecHitMixer);
DEFINE_FWK_MODULE(HORecHitMixer);
//DEFINE_FWK_MODULE(ZDCRecHitMixer);
DEFINE_FWK_MODULE(CastorRecHitMixer);



