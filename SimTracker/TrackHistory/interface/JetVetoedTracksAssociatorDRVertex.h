
#ifndef JetVetoedTracksAssociationDRVertex_h
#define JetVetoedTracksAssociationDRVertex_h

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "SimTracker/TrackHistory/interface/TrackClassifier.h"

class JetVetoedTracksAssociationDRVertex
{

public:

    JetVetoedTracksAssociationDRVertex (double fDr);
    ~JetVetoedTracksAssociationDRVertex () {}

    void produce (
        reco::JetTracksAssociation::Container* fAssociation,
        const std::vector<edm::RefToBase<reco::Jet> >& fJets,
        const std::vector<reco::TrackRef>& fTracks,
        TrackClassifier & classifier
    ) const;

private:
    /// fidutial dR between track in the vertex and jet's reference direction
    double mDeltaR2Threshold;
};

#endif
