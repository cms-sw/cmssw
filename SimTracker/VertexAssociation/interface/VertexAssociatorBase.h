#ifndef SimTracker_VertexAssociatorBase_h
#define SimTracker_VertexAssociatorBase_h

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OneToManyWithQuality.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "FWCore/Framework/interface/Event.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"

#include "SimTracker/TrackAssociation/interface/TrackAssociatorBase.h"

namespace reco
{
typedef edm::AssociationMap<edm::OneToManyWithQuality <TrackingVertexCollection, reco::VertexCollection, double> >
VertexSimToRecoCollection;
typedef edm::AssociationMap<edm::OneToManyWithQuality <reco::VertexCollection, TrackingVertexCollection, double> >
VertexRecoToSimCollection;
}

class VertexAssociatorBase
{
public:
    VertexAssociatorBase()
    {
        ;
    }
    virtual ~VertexAssociatorBase()
    {
        ;
    }

    virtual reco::VertexRecoToSimCollection
    associateRecoToSim (edm::Handle<reco::VertexCollection>& vc,
                        edm::Handle<TrackingVertexCollection>& tvc,
                        const edm::Event&    event,
                        reco::RecoToSimCollection& trackAssocResult) const = 0;

    virtual reco::VertexSimToRecoCollection
    associateSimToReco (edm::Handle<reco::VertexCollection>& vc,
                        edm::Handle<TrackingVertexCollection>& tvc ,
                        const edm::Event& event,
                        reco::SimToRecoCollection& trackAssocResult) const = 0;
};

#endif
