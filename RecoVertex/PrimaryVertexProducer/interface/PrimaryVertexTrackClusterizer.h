#ifndef PrimaryVertexTrackClusterizer_h
#define PrimaryVertexTrackClusterizer_h

/**\class PrimaryVertexTrackClusterizer
 
  Description: interface/base class for track clusterizers that separate event tracks into clusters along the beam line
  extends TrackClusterizerInZ

*/

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <vector>
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/PrimaryVertexProducer/interface/TrackClusterizerInZ.h"

class PrimaryVertexTrackClusterizer : public TrackClusterizerInZ {
public:
  PrimaryVertexTrackClusterizer() = default;
  PrimaryVertexTrackClusterizer(const edm::ParameterSet& conf){};
  virtual std::vector<TransientVertex> vertices(const std::vector<reco::TransientTrack>& tracks) const = 0;
  virtual std::vector<std::vector<reco::TransientTrack> > clusterize(
      const std::vector<reco::TransientTrack>& tracks) const = 0;

  virtual ~PrimaryVertexTrackClusterizer() = default;
};

#endif
