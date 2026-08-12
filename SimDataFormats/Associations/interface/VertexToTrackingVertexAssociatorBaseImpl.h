#ifndef SimDataFormats_Associations_VertexToTrackingVertexAssociatorBaseImpl_h
#define SimDataFormats_Associations_VertexToTrackingVertexAssociatorBaseImpl_h

#include "DataFormats/Common/interface/Handle.h"
#include "SimDataFormats/Associations/interface/VertexAssociation.h"

namespace reco {
  template <typename VertexCollection>
  class VertexToTrackingVertexAssociatorBaseImpl {
  public:
    using VertexType = typename VertexCollection::value_type;
    // association maps for templated Vertex <-> TrackingVertex
    using SimToRecoCollection =
        edm::AssociationMap<edm::OneToManyWithQuality<TrackingVertexCollection, edm::View<VertexType>, double>>;
    using RecoToSimCollection =
        edm::AssociationMap<edm::OneToManyWithQuality<edm::View<VertexType>, TrackingVertexCollection, double>>;

    /// Constructor
    VertexToTrackingVertexAssociatorBaseImpl();
    /// Destructor
    virtual ~VertexToTrackingVertexAssociatorBaseImpl();

    /// compare reco to sim the handle of Vertex and TrackingVertex
    /// collections
    virtual RecoToSimCollection associateRecoToSim(const edm::Handle<edm::View<VertexType>> &,
                                                   const edm::Handle<TrackingVertexCollection> &) const = 0;

    /// compare sim to reco the handle of Vertex and TrackingVertex
    /// collections
    virtual SimToRecoCollection associateSimToReco(const edm::Handle<edm::View<VertexType>> &,
                                                   const edm::Handle<TrackingVertexCollection> &) const = 0;
  };
}  // namespace reco

#endif
