#include "TauAnalysis/MCEmbeddingTools/plugins/CaloRecHitMixer.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HORecHit.h"
#include "DataFormats/HcalRecHit/interface/ZDCRecHit.h"
#include "DataFormats/HcalRecHit/interface/CastorRecHit.h"

typedef CaloRecHitMixer<EcalRecHit> EcalRHMixer;
typedef CaloRecHitMixer<HBHERecHit> HBHERHMixer;
typedef CaloRecHitMixer<HFRecHit> HFRHMixer;
typedef CaloRecHitMixer<HORecHit> HORHMixer;
#warning "ZDCRHMixer still needs to be done" 
//typedef CaloRecHitMixer<ZDCRecHit> ZDCRHMixer;
typedef CaloRecHitMixer<CastorRecHit> CastorRHMixer;

//-------------------------------------------------------------------------------
// define 'cleanRH' functions used for different types of recHits
//-------------------------------------------------------------------------------

template <typename T> 
T CaloRecHitMixer<T>::cleanRH(const T& rh, double muonEnergyDeposit)
{
  assert(0); // CV: make sure general function never gets called;
             //     always use template specializations
}

// template specialization for ECAL recHits
template <>
EcalRecHit CaloRecHitMixer<EcalRecHit>::cleanRH(const EcalRecHit& rh, double muonEnergyDeposit)
{
  double cleanedRecHitEnergy = rh.energy() - muonEnergyDeposit;
  EcalRecHit cleanedRecHit(rh.detid(), cleanedRecHitEnergy, rh.time(), rh.flags(), rh.checkFlagMask(0xFFFF));
  return cleanedRecHit;
}

// template specialization for different types of HCAL recHits
namespace
{
  template <typename T>
  T cleanRH_HCAL(const T& rh, double muonEnergyDeposit)
  {
    double cleanedRecHitEnergy = rh.energy() - muonEnergyDeposit;
    T cleanedRecHit(rh.detid(), cleanedRecHitEnergy, rh.time());
  
    cleanedRecHit.setFlags(rh.flags());
    
    cleanedRecHit.setAux(rh.aux()); // TF: aux does not seem to be used anywere (LXR search)

    return cleanedRecHit;
  }
}

template <>
HBHERecHit CaloRecHitMixer<HBHERecHit>::cleanRH(const HBHERecHit& rh, double muonEnergyDeposit)
{
  return cleanRH_HCAL(rh, muonEnergyDeposit);
}

template <>
HORecHit CaloRecHitMixer<HORecHit>::cleanRH(const HORecHit& rh, double muonEnergyDeposit)
{
  return cleanRH_HCAL(rh, muonEnergyDeposit);
}

template <>
HFRecHit CaloRecHitMixer<HFRecHit>::cleanRH(const HFRecHit& rh, double muonEnergyDeposit)
{
  return cleanRH_HCAL(rh, muonEnergyDeposit);
}

// template specialization for CASTOR recHits
template <>
CastorRecHit CaloRecHitMixer<CastorRecHit>::cleanRH(const CastorRecHit& rh, double muonEnergyDeposit)
{
  double cleanedRecHitEnergy = rh.energy() - muonEnergyDeposit;
  CastorRecHit cleanedRecHit(rh.detid(), cleanedRecHitEnergy, rh.time());
  return cleanedRecHit;
}

//-------------------------------------------------------------------------------
// define 'mergeRH' functions used for different types of recHits
//-------------------------------------------------------------------------------

template <typename T>
T CaloRecHitMixer<T>::mergeRH(const T& rh1, const T& rh2)
{
  assert(0); // CV: make sure general function never gets called;
             //     always use template specializations
}

// template specialization for ECAL recHits
//
// NOTE: rh1 should contain Zmumu part, rh2 - tau tau 
//
template <>
EcalRecHit CaloRecHitMixer<EcalRecHit>::mergeRH(const EcalRecHit& rh1, const EcalRecHit& rh2)
{
  // TF: is time calculation good this way ?
  EcalRecHit mergedRecHit(rh1.detid(), rh1.energy() + rh2.energy(), rh2.time(), rh2.flags(), rh1.checkFlagMask(0xFFFF));
  return mergedRecHit;
}

// template specialization for different types of HCAL recHits
//
// NOTE: rh1 should contain Zmumu part, rh2 - tau tau 
//
namespace
{
  template <typename T>
  T mergeRH_HCAL(const T& rh1, const T& rh2)
  {
    // TF: is time calculation good this way ?
    T mergedRecHit(rh1.detid(), rh1.energy() + rh2.energy(), rh2.time());

    // TF: take flags from Zmumu (data), since it is more likely to show problems than MC
    mergedRecHit.setFlags(rh1.flags());

    mergedRecHit.setAux(rh2.aux()); // TF: aux does not seem to be used anywere (LXR search)

    return mergedRecHit;
  }
}

template <>
HBHERecHit CaloRecHitMixer<HBHERecHit>::mergeRH(const HBHERecHit& rh1, const HBHERecHit& rh2)
{
  return mergeRH_HCAL(rh1, rh2);
}

template <>
HORecHit CaloRecHitMixer<HORecHit>::mergeRH(const HORecHit& rh1, const HORecHit& rh2)
{
  return mergeRH_HCAL(rh1, rh2);
}

template <>
HFRecHit CaloRecHitMixer<HFRecHit>::mergeRH(const HFRecHit& rh1, const HFRecHit& rh2)
{
  return mergeRH_HCAL(rh1, rh2);
}

// template specialization for CASTOR recHits
template <>
CastorRecHit CaloRecHitMixer<CastorRecHit>::mergeRH(const CastorRecHit& rh1, const CastorRecHit& rh2)
{
  // TF: is time calculation good this way ?
  CastorRecHit mergedRecHit(rh1.detid(), rh1.energy() + rh2.energy(), rh2.time());
  return mergedRecHit;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(EcalRHMixer);
DEFINE_FWK_MODULE(HBHERHMixer);
DEFINE_FWK_MODULE(HFRHMixer);
DEFINE_FWK_MODULE(HORHMixer);
//DEFINE_FWK_MODULE(ZDCRHMixer);
DEFINE_FWK_MODULE(CastorRHMixer);



