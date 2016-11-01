#include "TauAnalysis/MCEmbeddingTools/plugins/CaloCleaner.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HORecHit.h"
#include "DataFormats/HcalRecHit/interface/ZDCRecHit.h"
#include "DataFormats/HcalRecHit/interface/CastorRecHit.h"

typedef CaloCleaner<EcalRecHit> EcalRecHitCleaner;
typedef CaloCleaner<HBHERecHit> HBHERecHitCleaner;
typedef CaloCleaner<HFRecHit> HFRecHitCleaner;
typedef CaloCleaner<HORecHit> HORecHitCleaner;
typedef CaloCleaner<CastorRecHit> CastorRecHitCleaner;
typedef CaloCleaner<ZDCRecHit> ZDCRecHitCleaner;


//-------------------------------------------------------------------------------
// define 'buildRecHit' functions used for different types of recHits
//-------------------------------------------------------------------------------
  


template <typename T>
void  CaloCleaner<T>::fill_correction_map(TrackDetMatchInfo *,  std::map<uint32_t, float> *)
{
  assert(0); // CV: make sure general function never gets called;
             //     always use template specializations
}

template <>
void  CaloCleaner<EcalRecHit>::fill_correction_map(TrackDetMatchInfo * info,  std::map<uint32_t, float> * cor_map)
{
  if (is_preshower_){
     for ( std::vector<DetId>::const_iterator detId = info->crossedPreshowerIds.begin(); detId != info->crossedPreshowerIds.end(); ++detId ) {
       (*cor_map) [detId->rawId()] = 9999999; // just remove all energy (Below 0 is not possible)  
      }
    } 
  else {
    for(std::vector<const EcalRecHit*>::const_iterator hit=info->crossedEcalRecHits.begin(); hit!=info->crossedEcalRecHits.end(); hit++){
      //    (*cor_map) [(*hit)->detid().rawId()] +=(*hit)->energy();
      (*cor_map) [(*hit)->detid().rawId()] =(*hit)->energy(); 
    }
  }
}


template <>
void  CaloCleaner<HBHERecHit>::fill_correction_map(TrackDetMatchInfo * info,  std::map<uint32_t, float> * cor_map)
{
  for(std::vector<const HBHERecHit*>::const_iterator hit = info->crossedHcalRecHits.begin(); hit != info->crossedHcalRecHits.end(); hit++) {
    (*cor_map) [(*hit)->detid().rawId()] =(*hit)->energy(); 
  }
}


template <>
void  CaloCleaner<HORecHit>::fill_correction_map(TrackDetMatchInfo * info,  std::map<uint32_t, float> * cor_map)
{
  for(std::vector<const HORecHit*>::const_iterator hit = info->crossedHORecHits.begin(); hit != info->crossedHORecHits.end(); hit++) {
    (*cor_map) [(*hit)->detid().rawId()] =(*hit)->energy(); 
  }
}


template <>
void  CaloCleaner<HFRecHit>::fill_correction_map(TrackDetMatchInfo * info,  std::map<uint32_t, float> * cor_map)
{
 return; // No corrections for HF 
}

template <>
void  CaloCleaner<CastorRecHit>::fill_correction_map(TrackDetMatchInfo * info,  std::map<uint32_t, float> * cor_map)
{
 return;// No corrections for Castor
}

template <>
void  CaloCleaner<ZDCRecHit>::fill_correction_map(TrackDetMatchInfo * info,  std::map<uint32_t, float> * cor_map)
{
 return;// No corrections for Castor
}





#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(EcalRecHitCleaner);
DEFINE_FWK_MODULE(HBHERecHitCleaner);
DEFINE_FWK_MODULE(HORecHitCleaner);
// no  need for cleaning outside of tracker, so just a copy of the old collection
DEFINE_FWK_MODULE(HFRecHitCleaner);
DEFINE_FWK_MODULE(CastorRecHitCleaner);
DEFINE_FWK_MODULE(ZDCRecHitCleaner);



