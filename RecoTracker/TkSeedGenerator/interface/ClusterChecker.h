#ifndef RecoTracker_TkSeedGenerator_ClusterChecker_H
#define RecoTracker_TkSeedGenerator_ClusterChecker_H

#include "FWCore/Utilities/interface/InputTag.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
#include "FWCore/Framework/interface/ConsumesCollector.h"
#else
namespace edm {
	class ConsumesCollector;
}
#endif

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
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
  ClusterChecker(const edm::ParameterSet & conf, edm::ConsumesCollector & iC) ;
  ClusterChecker(const edm::ParameterSet & conf, edm::ConsumesCollector && iC) ;
#endif

  ~ClusterChecker() ;
  size_t tooManyClusters(const edm::Event & e) const ;

 private: 
  ClusterChecker(); // This is only needed for StringCutObjectSelector
  bool doACheck_;
  edm::InputTag clusterCollectionInputTag_;
  edm::InputTag pixelClusterCollectionInputTag_;
  unsigned int maxNrOfCosmicClusters_;
  unsigned int maxNrOfPixelClusters_;
  StringCutObjectSelector<reco::utils::ClusterTotals> selector_;
  unsigned int ignoreDetsAboveNClusters_;
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
  edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster> > token_sc;
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > token_pc;
#endif
};

#endif
