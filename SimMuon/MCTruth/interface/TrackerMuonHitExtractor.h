//
// modified & integrated by Giovanni Abbiendi
// from code by Arun Luthra:
// UserCode/luthra/MuonTrackSelector/src/MuonTrackSelector.cc
//
#ifndef MCTruth_TrackerMuonHitExtractor_h
#define MCTruth_TrackerMuonHitExtractor_h

#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <memory>

class TrackerMuonHitExtractor {
public:
  explicit TrackerMuonHitExtractor(const edm::ParameterSet &, edm::ConsumesCollector &&ic);
  ~TrackerMuonHitExtractor() = default;

  void init(const edm::Event &);
  std::vector<const TrackingRecHit *> getMuonHits(const reco::Muon &mu) const;

private:
  const edm::EDGetTokenT<DTRecSegment4DCollection> inputDTRecSegment4DToken_;
  const edm::EDGetTokenT<CSCSegmentCollection> inputCSCSegmentToken_;
};

#endif
