#ifndef Validation_CSCRecHits_CSCRecHitMatcher_h
#define Validation_CSCRecHits_CSCRecHitMatcher_h

/**\class DigiMatcher

 Description: Matching of rechits and segments for SimTrack in CSC

 Author:  Sven Dildick
*/

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "Validation/MuonCSCDigis/interface/CSCDigiMatcher.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"

#include <vector>
#include <map>
#include <set>

typedef std::vector<CSCRecHit2D> CSCRecHit2DContainer;
typedef std::vector<CSCSegment> CSCSegmentContainer;

class CSCRecHitMatcher {
public:
  // constructor
  CSCRecHitMatcher(edm::ParameterSet const& iPS, edm::ConsumesCollector&& iC);

  // destructor
  ~CSCRecHitMatcher() {}

  // initialize the event
  void init(const edm::Event& e, const edm::EventSetup& eventSetup);

  // do the matching
  void match(const SimTrack& t, const SimVertex& v);

  // layer detIds with CSCRecHit2D
  std::set<unsigned int> layerIdsCSCRecHit2D() const;
  // chamber detIds with CSCRecHit2D
  std::set<unsigned int> chamberIdsCSCRecHit2D() const;
  // chamber detIds with CSCSegment
  std::set<unsigned int> chamberIdsCSCSegment() const;

  //CSC rechits from a particular layer or chamber
  const CSCRecHit2DContainer& cscRecHit2DsInLayer(unsigned int) const;
  const CSCRecHit2DContainer& cscRecHit2DsInChamber(unsigned int) const;
  //CSC segments from a particular chamber
  const CSCSegmentContainer& cscSegmentsInChamber(unsigned int) const;

  const CSCSegmentContainer cscSegments() const;
  const CSCRecHit2DContainer cscRecHit2Ds() const;

  // check if a certain rechit appears in a container
  bool cscRecHit2DInContainer(const CSCRecHit2D&, const CSCRecHit2DContainer&) const;
  bool cscSegmentInContainer(const CSCSegment&, const CSCSegmentContainer&) const;

  // check if a certain rechit was matched to a simtrack
  bool isCSCRecHit2DMatched(const CSCRecHit2D&) const;
  bool isCSCSegmentMatched(const CSCSegment&) const;

  int nCSCRecHit2Ds() const;
  int nCSCSegments() const;
  bool areCSCSegmentsSame(const CSCSegment&, const CSCSegment&) const;
  bool areCSCRecHit2DsSame(const CSCRecHit2D&, const CSCRecHit2D&) const;

  int nCSCRecHit2DsInLayer(unsigned int) const;
  int nCSCRecHit2DsInChamber(unsigned int) const;
  int nCSCSegmentsInChamber(unsigned int) const;

  CSCSegment bestCSCSegment(unsigned int);

  GlobalPoint globalPoint(const CSCSegment&) const;

private:
  std::unique_ptr<CSCDigiMatcher> cscDigiMatcher_;

  edm::EDGetTokenT<CSCRecHit2DCollection> cscRecHit2DToken_;
  edm::EDGetTokenT<CSCSegmentCollection> cscSegmentToken_;

  edm::Handle<CSCRecHit2DCollection> cscRecHit2DH_;
  edm::Handle<CSCSegmentCollection> cscSegmentH_;

  edm::ESHandle<CSCGeometry> csc_geom_;
  const CSCGeometry* cscGeometry_;

  void matchCSCRecHit2DsToSimTrack(const CSCRecHit2DCollection&);
  void matchCSCSegmentsToSimTrack(const CSCSegmentCollection&);

  int verboseCSCRecHit2D_;
  int maxBXCSCRecHit2D_;
  int minBXCSCRecHit2D_;

  int verboseCSCSegment_;
  int maxBXCSCSegment_;
  int minBXCSCSegment_;

  std::map<unsigned int, CSCRecHit2DContainer> layer_to_cscRecHit2D_;
  std::map<unsigned int, CSCRecHit2DContainer> chamber_to_cscRecHit2D_;
  std::map<unsigned int, CSCSegmentContainer> chamber_to_cscSegment_;

  CSCRecHit2DContainer no_cscRecHit2Ds_;
  CSCSegmentContainer no_cscSegments_;
};

#endif
