#include "TauAnalysis/MCEmbeddingTools/plugins/TrackerCleaner.h"

#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"

typedef TrackerCleaner<SiPixelCluster> PixelColCleaner;
typedef TrackerCleaner<SiStripCluster> StripColCleaner;

//-------------------------------------------------------------------------------
// define 'buildRecHit' functions used for different types of recHits
//-------------------------------------------------------------------------------

template <typename T>
bool TrackerCleaner<T>::match_rechit_type(const TrackingRecHit &murechit) {
  assert(0);  // CV: make sure general function never gets called;
              //     always use template specializations
  return false;
}

template <>
bool TrackerCleaner<SiStripCluster>::match_rechit_type(const TrackingRecHit &murechit) {
  const std::type_info &hit_type = typeid(murechit);
  if (hit_type == typeid(SiStripRecHit2D))
    return true;
  else if (hit_type == typeid(SiStripRecHit1D))
    return true;
  else if (hit_type == typeid(SiStripMatchedRecHit2D))
    return true;
  else if (hit_type == typeid(ProjectedSiStripRecHit2D))
    return true;

  return false;
}

template <>
bool TrackerCleaner<SiPixelCluster>::match_rechit_type(const TrackingRecHit &murechit) {
  const std::type_info &hit_type = typeid(murechit);
  if (hit_type == typeid(SiPixelRecHit))
    return true;

  return false;
}

DEFINE_FWK_MODULE(PixelColCleaner);
DEFINE_FWK_MODULE(StripColCleaner);
