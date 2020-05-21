#include "TauAnalysis/MCEmbeddingTools/plugins/CaloCleaner.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HORecHit.h"
#include "DataFormats/HcalRecHit/interface/ZDCRecHit.h"
#include "DataFormats/HcalRecHit/interface/CastorRecHit.h"

typedef CaloCleaner<EcalRecHit> EcalRecHitColCleaner;
typedef CaloCleaner<HBHERecHit> HBHERecHitColCleaner;
typedef CaloCleaner<HFRecHit> HFRecHitColCleaner;
typedef CaloCleaner<HORecHit> HORecHitColCleaner;
typedef CaloCleaner<CastorRecHit> CastorRecHitColCleaner;
typedef CaloCleaner<ZDCRecHit> ZDCRecHitColCleaner;

//-------------------------------------------------------------------------------
// define 'buildRecHit' functions used for different types of recHits
//-------------------------------------------------------------------------------

template <typename T>
void CaloCleaner<T>::fill_correction_map(TrackDetMatchInfo *, std::map<uint32_t, float> *) {
  assert(0);  // CV: make sure general function never gets called;
              //     always use template specializations
}

template <>
void CaloCleaner<EcalRecHit>::fill_correction_map(TrackDetMatchInfo *info, std::map<uint32_t, float> *cor_map) {
  if (is_preshower_) {
    for (auto crossedPreshowerId : info->crossedPreshowerIds) {
      (*cor_map)[crossedPreshowerId.rawId()] = 9999999;  // just remove all energy (Below 0 is not possible)
    }
  } else {
    for (auto crossedEcalRecHit : info->crossedEcalRecHits) {
      //    (*cor_map) [(*hit)->detid().rawId()] +=(*hit)->energy();
      (*cor_map)[crossedEcalRecHit->detid().rawId()] = crossedEcalRecHit->energy();
    }
  }
}

template <>
void CaloCleaner<HBHERecHit>::fill_correction_map(TrackDetMatchInfo *info, std::map<uint32_t, float> *cor_map) {
  for (auto crossedHcalRecHit : info->crossedHcalRecHits) {
    (*cor_map)[crossedHcalRecHit->detid().rawId()] = crossedHcalRecHit->energy();
  }
}

template <>
void CaloCleaner<HORecHit>::fill_correction_map(TrackDetMatchInfo *info, std::map<uint32_t, float> *cor_map) {
  for (auto crossedHORecHit : info->crossedHORecHits) {
    (*cor_map)[crossedHORecHit->detid().rawId()] = crossedHORecHit->energy();
  }
}

template <>
void CaloCleaner<HFRecHit>::fill_correction_map(TrackDetMatchInfo *info, std::map<uint32_t, float> *cor_map) {
  return;  // No corrections for HF
}

template <>
void CaloCleaner<CastorRecHit>::fill_correction_map(TrackDetMatchInfo *info, std::map<uint32_t, float> *cor_map) {
  return;  // No corrections for Castor
}

template <>
void CaloCleaner<ZDCRecHit>::fill_correction_map(TrackDetMatchInfo *info, std::map<uint32_t, float> *cor_map) {
  return;  // No corrections for Castor
}

DEFINE_FWK_MODULE(EcalRecHitColCleaner);
DEFINE_FWK_MODULE(HBHERecHitColCleaner);
DEFINE_FWK_MODULE(HORecHitColCleaner);
// no  need for cleaning outside of tracker, so just a copy of the old collection
DEFINE_FWK_MODULE(HFRecHitColCleaner);
DEFINE_FWK_MODULE(CastorRecHitColCleaner);
DEFINE_FWK_MODULE(ZDCRecHitColCleaner);
