#ifndef VertexAssociatorByTracks_h
#define VertexAssociatorByTracks_h

#include "SimGeneral/TrackingAnalysis/interface/TrackingParticleSelector.h"

#include "DataFormats/TrackReco/interface/Track.h"

#include "SimTracker/VertexAssociation/interface/VertexAssociatorBase.h"


class VertexAssociatorByTracks : public VertexAssociatorBase
{

public:

    explicit VertexAssociatorByTracks( const edm::ParameterSet& );
    ~VertexAssociatorByTracks();

    /* Associate TrackingVertex to RecoVertex By Hits */

    reco::VertexRecoToSimCollection
    associateRecoToSim (
        edm::Handle<edm::View<reco::Vertex> >&,
        edm::Handle<TrackingVertexCollection> &,
        const edm::Event &,
        reco::RecoToSimCollection &
    ) const;

    reco::VertexSimToRecoCollection
    associateSimToReco (
        edm::Handle<edm::View<reco::Vertex> >&,
        edm::Handle<TrackingVertexCollection> &,
        const edm::Event &,
        reco::SimToRecoCollection &
    ) const;

private:

    // ----- member data
    const edm::ParameterSet & config_;

    double R2SMatchedSimRatio_;
    double R2SMatchedRecoRatio_;
    double S2RMatchedSimRatio_;
    double S2RMatchedRecoRatio_;

    TrackingParticleSelector selector_;
    reco::TrackBase::TrackQuality trackQuality_;

};

#endif
