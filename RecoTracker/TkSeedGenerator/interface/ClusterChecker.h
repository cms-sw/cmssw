#ifndef RecoTracker_TkSeedGenerator_ClusterChecker_H
#define RecoTracker_TkSeedGenerator_ClusterChecker_H

#include "FWCore/Utilities/interface/InputTag.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"

namespace edm {
  class Event;
  class ParameterSet;
  class ParameterSetDescription;
}  // namespace edm

namespace reco {
  namespace utils {
    struct ClusterTotals {
      ClusterTotals() : strip(0), pixel(0), stripdets(0), pixeldets(0) {}
      int strip;      /// number of strip clusters
      int pixel;      /// number of pixel clusters
      int stripdets;  /// number of strip detectors with at least one cluster
      int pixeldets;  /// number of pixel detectors with at least one cluster
    };
  }  // namespace utils
}  // namespace reco

class ClusterChecker {
public:
  ClusterChecker(const edm::ParameterSet& conf, edm::ConsumesCollector& iC);
  ClusterChecker(const edm::ParameterSet& conf, edm::ConsumesCollector&& iC);
  ClusterChecker() = delete;  // This is only needed for StringCutObjectSelector

  static void fillDescriptions(edm::ParameterSetDescription& description);

  ~ClusterChecker();
  size_t tooManyClusters(const edm::Event& e) const;

private:
  bool doACheck_;
  edm::InputTag clusterCollectionInputTag_;
  edm::InputTag pixelClusterCollectionInputTag_;
  unsigned int maxNrOfStripClusters_;
  unsigned int maxNrOfPixelClusters_;
  StringCutObjectSelector<reco::utils::ClusterTotals> selector_;
  unsigned int ignoreDetsAboveNClusters_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster> > token_sc;
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > token_pc;
};

#endif
