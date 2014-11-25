
#ifndef RecoMuon_ME0MuonTrackCollProducer_h
#define RecoMuon_ME0MuonTrackCollProducer_h

#include <memory>
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/ME0Muon.h"
#include "DataFormats/MuonReco/interface/ME0MuonCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

class ME0MuonTrackCollProducer : public edm::EDProducer {
  public:
    explicit ME0MuonTrackCollProducer(const edm::ParameterSet&);
    //std::vector<double> findSimVtx(edm::Event& iEvent);
    ~ME0MuonTrackCollProducer();

  private:
    virtual void produce(edm::Event&, const edm::EventSetup&);
    edm::Handle <std::vector<reco::ME0Muon> > OurMuons;
    //edm::Handle<reco::ME0MuonCollection> muonCollectionH;
    edm::InputTag muonsTag;
    edm::InputTag vxtTag;
    bool useIPxy, useIPz;
    std::vector<std::string> selectionTags;
    std::string trackType;
    const edm::ParameterSet parset_;
};

#endif
