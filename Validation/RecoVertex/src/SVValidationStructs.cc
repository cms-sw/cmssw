#include "Validation/RecoVertex/interface/SVValidationStructs.h"

template <>
edm::RefToBase<reco::Vertex> RecoSecondaryVertex::recoVertex<reco::Vertex>() const {
  return recoVertexRef.value();
}

template <>
edm::RefToBase<reco::VertexCompositePtrCandidate> RecoSecondaryVertex::recoVertex<reco::VertexCompositePtrCandidate>()
    const {
  return recoVertexCPCRef.value();
}
