#ifndef SimTracker_VertexAssociatorBase_h
#define SimTracker_VertexAssociatorBase_h

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OneToManyWithQuality.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/RecoCandidate/interface/TrackAssociation.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "FWCore/Framework/interface/Event.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
#include "SimDataFormats/Associations/interface/VertexAssociation.h"


namespace reco
{
    typedef edm::RefToBase<reco::Vertex> VertexBaseRef;
}

class VertexAssociatorBase
{

public:

    VertexAssociatorBase(){}
    virtual ~VertexAssociatorBase(){}

    virtual reco::VertexRecoToSimCollection
    associateRecoToSim (edm::Handle<edm::View<reco::Vertex> >&,
                        edm::Handle<TrackingVertexCollection>&,
                        const edm::Event&,
                        reco::RecoToSimCollection&) const = 0;

    virtual reco::VertexSimToRecoCollection
    associateSimToReco (edm::Handle<edm::View<reco::Vertex> >&,
                        edm::Handle<TrackingVertexCollection>&,
                        const edm::Event&,
                        reco::SimToRecoCollection&) const = 0;
};

#endif
