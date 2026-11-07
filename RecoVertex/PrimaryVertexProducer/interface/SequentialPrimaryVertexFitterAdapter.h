#ifndef SequentialPrimaryVertexFitterAdapter_h
#define SequentialPrimaryVertexFitterAdapter_h

/**\class SequentialPrimaryVertexFitterAdapter

  Description: Adapter class for Kalman and Adaptive vertex fitters

*/

#include <map>
#include <sstream>

#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoVertex/PrimaryVertexProducer/interface/PrimaryVertexFitterBase.h"
#include "RecoVertex/VertexPrimitives/interface/VertexFitter.h"

class SequentialPrimaryVertexFitterAdapter : public PrimaryVertexFitterBase {
public:
  SequentialPrimaryVertexFitterAdapter() : fitter(nullptr) {}
  SequentialPrimaryVertexFitterAdapter(const VertexFitter<5>* vertex_fitter, bool useClusterWeights = false)
      : fitter(vertex_fitter), useClusterWeights_(useClusterWeights) {}
  ~SequentialPrimaryVertexFitterAdapter() override = default;

  std::vector<TransientVertex> fit(const std::vector<reco::TransientTrack>& dummy,
                                   const std::vector<TransientVertex>& clusters,
                                   const reco::BeamSpot& beamspot,
                                   const bool useBeamConstraint) override {
    std::vector<TransientVertex> pvs;
    for (auto& cluster : clusters) {
      const std::vector<reco::TransientTrack>& tracklist = cluster.originalTracks();
      TransientVertex v;
      if (useBeamConstraint && (tracklist.size() > 1)) {
        try {
          v = fitter->vertex(tracklist, beamspot);
        } catch (VertexException& ex) {
          std::ostringstream beamspotInfo;
          beamspotInfo << "While processing SequentialPrimaryVertexFitterAdapter::fit() with BeamSpot parameters: \n"
                       << beamspot;
          ex.addContext(beamspotInfo.str());
          throw;  // rethrow the exception
        }
      } else if (!(useBeamConstraint) && (tracklist.size() > 1)) {
        v = fitter->vertex(tracklist);
      }  // else: no fit ==> v.isValid()=False

      if (v.isValid()) {
        if (useClusterWeights_ && cluster.hasTrackWeight()) {
          std::map<const reco::Track*, float> clusterWeight;
          for (const auto& kv : cluster.weightMap()) {
            clusterWeight[&(kv.first.track())] = kv.second;
          }
          TransientVertex::TransientTrackToFloatMap weights;
          for (const auto& tt : v.originalTracks()) {
            auto it = clusterWeight.find(&(tt.track()));
            weights[tt] = (it != clusterWeight.end()) ? it->second : 1.0f;
          }
          v.weightMap(weights);
        }
        pvs.push_back(v);
      }
    }
    return pvs;
  };

protected:
  // configuration
  const VertexFitter<5>* fitter;  // Kalman or Adaptive
  bool useClusterWeights_ = false;
};
#endif
