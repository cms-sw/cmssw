#ifndef SimDataFormats_Associations_VertexAssociation_h
#define SimDataFormats_Associations_VertexAssociation_h

#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/OneToManyWithQualityGeneric.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"

namespace reco {

  template <typename T_VertexColl>
  using VertexSimToRecoCollectionT =
      edm::AssociationMap<edm::OneToManyWithQuality<TrackingVertexCollection, T_VertexColl, double>>;

  using VertexSimToRecoCollection = VertexSimToRecoCollectionT<edm::View<reco::Vertex>>;
  using VertexSimToRecoCollectionCPC = VertexSimToRecoCollectionT<edm::View<reco::VertexCompositePtrCandidate>>;

  template <typename T_VertexColl>
  using VertexRecoToSimCollectionT =
      edm::AssociationMap<edm::OneToManyWithQuality<T_VertexColl, TrackingVertexCollection, double>>;

  using VertexRecoToSimCollection = VertexRecoToSimCollectionT<edm::View<reco::Vertex>>;
  using VertexRecoToSimCollectionCPC = VertexRecoToSimCollectionT<edm::View<reco::VertexCompositePtrCandidate>>;

}  // namespace reco

#endif
