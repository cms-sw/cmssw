//
// modified & integrated by Giovanni Abbiendi
// from code by Arun Luthra: UserCode/luthra/MuonTrackSelector/src/MuonTrackSelector.cc
//
#ifndef MCTruth_MuonTrackProducer_h
#define MCTruth_MuonTrackProducer_h

#include <memory>
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"

class MuonTrackProducer : public edm::stream::EDProducer<> {
  public:
    explicit MuonTrackProducer(const edm::ParameterSet&);
    virtual ~MuonTrackProducer();

  private:
    virtual void produce(edm::Event&, const edm::EventSetup&);
  
    edm::Handle<reco::MuonCollection> muonCollectionH;
    edm::Handle<DTRecSegment4DCollection> dtSegmentCollectionH_;
    edm::Handle<CSCSegmentCollection> cscSegmentCollectionH_;

    edm::EDGetTokenT<reco::MuonCollection> muonsToken;
    edm::EDGetTokenT<DTRecSegment4DCollection> inputDTRecSegment4DToken_;
    edm::EDGetTokenT<CSCSegmentCollection> inputCSCSegmentToken_;

    std::vector<std::string> selectionTags;
    std::string trackType;
    const edm::ParameterSet parset_;
};

#endif
