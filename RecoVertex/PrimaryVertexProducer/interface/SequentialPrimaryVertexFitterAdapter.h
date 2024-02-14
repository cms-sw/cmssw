#ifndef SequentialPrimaryVertexFitterAdapter_h
#define SequentialPrimaryVertexFitterAdapter_h

/**\class SequentialPrimaryVertexFitterAdapter
 
  Description: Adapter class for Kalman and Adaptive vertex fitters 

*/

#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoVertex/PrimaryVertexProducer/interface/PrimaryVertexFitterBase.h"
#include "RecoVertex/VertexPrimitives/interface/VertexFitter.h"

class SequentialPrimaryVertexFitterAdapter : public PrimaryVertexFitterBase {
public:
  SequentialPrimaryVertexFitterAdapter() : fitter(nullptr){};
  SequentialPrimaryVertexFitterAdapter(const VertexFitter<5>* vertex_fitter) : fitter(vertex_fitter){};
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
        v = fitter->vertex(tracklist, beamspot);
      } else if (!(useBeamConstraint) && (tracklist.size() > 1)) {
        v = fitter->vertex(tracklist);
      }  // else: no fit ==> v.isValid()=False

      if (v.isValid()) {
        pvs.push_back(v);
      }
    }
    return pvs;
  };

protected:
  // configuration
  const VertexFitter<5>* fitter;  // Kalman or Adaptive
};
#endif
