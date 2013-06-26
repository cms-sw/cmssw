//
// modified & integrated by Giovanni Abbiendi
// from code by Arun Luthra: UserCode/luthra/MuonTrackSelector/src/MuonTrackSelector.cc
//
#ifndef MCTruth_TrackerMuonHitExtractor_h
#define MCTruth_TrackerMuonHitExtractor_h

#include <memory>
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"

class TrackerMuonHitExtractor {
  public:
    explicit TrackerMuonHitExtractor(const edm::ParameterSet&);
    ~TrackerMuonHitExtractor();

    void init(const edm::Event&, const edm::EventSetup&);
    std::vector<const TrackingRecHit *> getMuonHits(const reco::Muon &mu) const ;
  private:
    edm::Handle<DTRecSegment4DCollection> dtSegmentCollectionH_;
    edm::Handle<CSCSegmentCollection> cscSegmentCollectionH_;

    edm::InputTag inputDTRecSegment4DCollection_;
    edm::InputTag inputCSCSegmentCollection_;
};

#endif
