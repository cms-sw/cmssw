#ifndef RecoTracker_TkTrackingRegions_VertexBeamspotOrigins_h
#define RecoTracker_TkTrackingRegions_VertexBeamspotOrigins_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include <vector>
#include <utility>

class VertexBeamspotOrigins {
public:
  using Origins = std::vector<std::pair<GlobalPoint, float> >;  // (origin, half-length in z)
  enum class OperationMode { BEAM_SPOT_FIXED, BEAM_SPOT_SIGMA, VERTICES_FIXED, VERTICES_SIGMA };

  VertexBeamspotOrigins(const edm::ParameterSet& regPSet, edm::ConsumesCollector&& iC)
      : VertexBeamspotOrigins(regPSet, iC) {}
  VertexBeamspotOrigins(const edm::ParameterSet& regPSet, edm::ConsumesCollector& iC);
  ~VertexBeamspotOrigins() = default;

  static void fillDescriptions(edm::ParameterSetDescription& desc,
                               const std::string& defaultBeamSpot = "offlineBeamSpot",
                               const std::string& defaultVertex = "firstStepPrimaryVertices",
                               int defaultMaxVertices = -1);

  Origins origins(const edm::Event& iEvent) const;

private:
  OperationMode m_operationMode;

  edm::EDGetTokenT<reco::VertexCollection> token_vertex;
  edm::EDGetTokenT<reco::BeamSpot> token_beamSpot;

  int m_maxNVertices;

  float m_nSigmaZBeamSpot;
  float m_zErrorBeamSpot;
  float m_nSigmaZVertex;
  float m_zErrorVertex;
};

#endif
