// \class JetVetoedTracksAssociatorAtVertex JetTracksAssociatorAtVertex.cc
//
// Original Author:  Andrea Rizzi
//         Created:  Wed Apr 12 11:12:49 CEST 2006
// Accommodated for Jet Package by: Fedor Ratnikov Jul. 30, 2007
// $Id: JetVetoedTracksAssociatorAtVertex.cc,v 1.4 2010/02/20 21:02:06 wmtan Exp $
//
//

#include "DataFormats/Common/interface/EDProductfwd.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimTracker/TrackHistory/interface/JetVetoedTracksAssociatorDRVertex.h"

class JetVetoedTracksAssociatorAtVertex : public edm::EDProducer
{
public:
    JetVetoedTracksAssociatorAtVertex(const edm::ParameterSet&);
    virtual ~JetVetoedTracksAssociatorAtVertex();
    virtual void produce(edm::Event&, const edm::EventSetup&);
private:
    edm::InputTag mJets;
    edm::InputTag mTracks;
    JetVetoedTracksAssociationDRVertex mAssociator;
    TrackClassifier classifier;
};

JetVetoedTracksAssociatorAtVertex::JetVetoedTracksAssociatorAtVertex(const edm::ParameterSet& fConfig)
        : mJets (fConfig.getParameter<edm::InputTag> ("jets")),
        mTracks (fConfig.getParameter<edm::InputTag> ("tracks")),
        mAssociator (fConfig.getParameter<double> ("coneSize")),
        classifier(fConfig)
{
    produces<reco::JetTracksAssociation::Container> ();
}

JetVetoedTracksAssociatorAtVertex::~JetVetoedTracksAssociatorAtVertex() {}

void JetVetoedTracksAssociatorAtVertex::produce(edm::Event& fEvent, const edm::EventSetup& fSetup)
{
    // Gather contextual information for TrackCategories
    classifier.newEvent(fEvent, fSetup);

    edm::Handle <edm::View <reco::Jet> > jets_h;
    fEvent.getByLabel (mJets, jets_h);
    edm::Handle <reco::TrackCollection> tracks_h;
    fEvent.getByLabel (mTracks, tracks_h);

    std::auto_ptr<reco::JetTracksAssociation::Container> jetTracks (new reco::JetTracksAssociation::Container (reco::JetRefBaseProd(jets_h)));

    // format inputs
    std::vector <edm::RefToBase<reco::Jet> > allJets;
    allJets.reserve (jets_h->size());
    for (unsigned i = 0; i < jets_h->size(); ++i) allJets.push_back (jets_h->refAt(i));
    std::vector <reco::TrackRef> allTracks;
    allTracks.reserve (tracks_h->size());
    for (unsigned i = 0; i < tracks_h->size(); ++i) allTracks.push_back (reco::TrackRef (tracks_h, i));
    // run algo
    mAssociator.produce (&*jetTracks, allJets, allTracks, classifier);
    // store output
    fEvent.put (jetTracks);
}

DEFINE_FWK_MODULE(JetVetoedTracksAssociatorAtVertex);

