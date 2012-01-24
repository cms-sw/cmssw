#ifndef RecoTracker_TkSeedGenerator_ClusterChecker_H
#define RecoTracker_TkSeedGenerator_ClusterChecker_H

#include "FWCore/Utilities/interface/InputTag.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

namespace edm { class Event; class ParameterSet; }

namespace reco { namespace utils {
    struct ClusterTotals {
       ClusterTotals() : strip(0), pixel(0), stripdets(0), pixeldets(0) {}
       int strip; /// number of strip clusters
       int pixel; /// number of pixel clusters
       int stripdets; /// number of strip detectors with at least one cluster
       int pixeldets; /// number of pixel detectors with at least one cluster    
    };
} }

class ClusterChecker {
 public: 
  ClusterChecker(const edm::ParameterSet & conf) ;
  ~ClusterChecker() ;
  size_t tooManyClusters(const edm::Event & e) const ;

 private: 
  bool doACheck_;
  edm::InputTag clusterCollectionInputTag_;
  edm::InputTag pixelClusterCollectionInputTag_;
  unsigned int maxNrOfCosmicClusters_;
  unsigned int maxNrOfPixelClusters_;
  StringCutObjectSelector<reco::utils::ClusterTotals> selector_;
  unsigned int ignoreDetsAboveNClusters_;
};

#endif
