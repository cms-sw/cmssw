#include <memory>

#include "Validation/CSCRecHits/interface/CSCRecHitMatcher.h"

using namespace std;

CSCRecHitMatcher::CSCRecHitMatcher(const edm::ParameterSet& pset, edm::ConsumesCollector&& iC) {
  const auto& cscRecHit2D = pset.getParameter<edm::ParameterSet>("cscRecHit");
  maxBXCSCRecHit2D_ = cscRecHit2D.getParameter<int>("maxBX");
  minBXCSCRecHit2D_ = cscRecHit2D.getParameter<int>("minBX");
  verboseCSCRecHit2D_ = cscRecHit2D.getParameter<int>("verbose");

  const auto& cscSegment = pset.getParameter<edm::ParameterSet>("cscSegment");
  maxBXCSCSegment_ = cscSegment.getParameter<int>("maxBX");
  minBXCSCSegment_ = cscSegment.getParameter<int>("minBX");
  verboseCSCSegment_ = cscSegment.getParameter<int>("verbose");

  // make a new digi matcher
  cscDigiMatcher_ = std::make_unique<CSCDigiMatcher>(pset, std::move(iC));

  cscRecHit2DToken_ = iC.consumes<CSCRecHit2DCollection>(cscRecHit2D.getParameter<edm::InputTag>("inputTag"));
  cscSegmentToken_ = iC.consumes<CSCSegmentCollection>(cscSegment.getParameter<edm::InputTag>("inputTag"));

  cscGeomToken_ = iC.esConsumes<CSCGeometry, MuonGeometryRecord>();
}

void CSCRecHitMatcher::init(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  cscDigiMatcher_->init(iEvent, iSetup);

  iEvent.getByToken(cscRecHit2DToken_, cscRecHit2DH_);
  iEvent.getByToken(cscSegmentToken_, cscSegmentH_);

  const auto& csc_geom = iSetup.getHandle(cscGeomToken_);
  if (csc_geom.isValid()) {
    cscGeometry_ = csc_geom.product();
  } else {
    edm::LogWarning("CSCSimHitMatcher") << "+++ Info: CSC geometry is unavailable. +++\n";
  }
}

/// do the matching
void CSCRecHitMatcher::match(const SimTrack& t, const SimVertex& v) {
  // match digis first
  cscDigiMatcher_->match(t, v);

  // get the rechit collection
  const CSCRecHit2DCollection& cscRecHit2Ds = *cscRecHit2DH_.product();
  const CSCSegmentCollection& cscSegments = *cscSegmentH_.product();

  // now match the rechits
  matchCSCRecHit2DsToSimTrack(cscRecHit2Ds);
  matchCSCSegmentsToSimTrack(cscSegments);
}

void CSCRecHitMatcher::matchCSCRecHit2DsToSimTrack(const CSCRecHit2DCollection& rechits) {
  if (verboseCSCRecHit2D_)
    edm::LogInfo("CSCRecHitMatcher") << "Matching simtrack to CSC rechits";

  // fetch all layerIds with digis
  const auto& strip_ids = cscDigiMatcher_->detIdsStrip();
  const auto& wire_ids = cscDigiMatcher_->detIdsWire();

  // merge the two collections
  std::set<unsigned int> layer_ids;
  layer_ids.insert(strip_ids.begin(), strip_ids.end());
  layer_ids.insert(wire_ids.begin(), wire_ids.end());

  if (verboseCSCRecHit2D_)
    edm::LogInfo("CSCRecHitMatcher") << "Number of matched csc layer_ids " << layer_ids.size();

  for (const auto& id : layer_ids) {
    CSCDetId p_id(id);

    // print all the wires in the CSCChamber
    const auto& hit_wg(cscDigiMatcher_->wiregroupsInDetId(id));
    if (verboseCSCRecHit2D_) {
      edm::LogInfo("CSCRecHitMatcher") << "hit wg csc from simhit" << endl;
      for (const auto& p : hit_wg) {
        edm::LogInfo("CSCRecHitMatcher") << p;
      }
    }

    // print all the strips in the CSCChamber
    const auto& hit_strips(cscDigiMatcher_->stripsInDetId(id));
    if (verboseCSCRecHit2D_) {
      edm::LogInfo("CSCRecHitMatcher") << "hit strip csc from simhit" << endl;
      for (const auto& p : hit_strips) {
        edm::LogInfo("CSCRecHitMatcher") << p;
      }
    }

    // get the rechits
    const auto& rechits_in_det = rechits.get(p_id);
    for (auto d = rechits_in_det.first; d != rechits_in_det.second; ++d) {
      if (verboseCSCRecHit2D_)
        edm::LogInfo("CSCRecHitMatcher") << "rechit " << p_id << " " << *d;

      // does the wire number match?
      const bool wireMatch(std::find(hit_wg.begin(), hit_wg.end(), d->hitWire()) != hit_wg.end());

      // does the strip number match?
      bool stripMatch(false);
      for (size_t iS = 0; iS < d->nStrips(); ++iS) {
        if (hit_strips.find(d->channels(iS)) != hit_strips.end())
          stripMatch = true;
      }

      // this rechit was matched to a matching simhit
      if (wireMatch and stripMatch) {
        if (verboseCSCRecHit2D_)
          edm::LogInfo("CSCRecHitMatcher") << "\t...was matched!";
        layer_to_cscRecHit2D_[id].push_back(*d);
        chamber_to_cscRecHit2D_[p_id.chamberId().rawId()].push_back(*d);
      }
    }
  }
}

void CSCRecHitMatcher::matchCSCSegmentsToSimTrack(const CSCSegmentCollection& cscSegments) {
  if (verboseCSCSegment_)
    edm::LogInfo("CSCRecHitMatcher") << "Matching simtrack to segments";
  // fetch all chamberIds with 2D rechits

  const auto& chamber_ids = chamberIdsCSCRecHit2D();
  if (verboseCSCSegment_)
    edm::LogInfo("CSCRecHitMatcher") << "Number of matched csc segments " << chamber_ids.size();
  for (const auto& id : chamber_ids) {
    CSCDetId p_id(id);

    // print all CSCRecHit2D in the CSCChamber
    const auto& csc_rechits(cscRecHit2DsInChamber(id));
    if (verboseCSCSegment_) {
      edm::LogInfo("CSCRecHitMatcher") << "hit csc rechits" << endl;
      for (const auto& p : csc_rechits) {
        edm::LogInfo("CSCRecHitMatcher") << p;
      }
    }

    // get the segments
    const auto& segments_in_det = cscSegments.get(p_id);
    for (auto d = segments_in_det.first; d != segments_in_det.second; ++d) {
      if (verboseCSCSegment_)
        edm::LogInfo("CSCRecHitMatcher") << "segment " << p_id << " " << *d << endl;

      //access the rechits
      const auto& recHits(d->recHits());

      int rechitsFound = 0;
      if (verboseCSCSegment_)
        edm::LogInfo("CSCRecHitMatcher") << recHits.size() << " csc rechits from segment " << endl;
      for (const auto& rh : recHits) {
        const CSCRecHit2D* cscrh(dynamic_cast<const CSCRecHit2D*>(rh));
        if (isCSCRecHit2DMatched(*cscrh))
          ++rechitsFound;
      }
      if (rechitsFound == 0)
        continue;
      if (verboseCSCSegment_) {
        edm::LogInfo("CSCRecHitMatcher") << "Found " << rechitsFound << " rechits out of "
                                         << cscRecHit2DsInChamber(id).size();
        edm::LogInfo("CSCRecHitMatcher") << "\t...was matched!";
      }
      chamber_to_cscSegment_[p_id.rawId()].push_back(*d);
    }
  }
}

std::set<unsigned int> CSCRecHitMatcher::layerIdsCSCRecHit2D() const {
  std::set<unsigned int> result;
  for (const auto& p : layer_to_cscRecHit2D_)
    result.insert(p.first);
  return result;
}

std::set<unsigned int> CSCRecHitMatcher::chamberIdsCSCRecHit2D() const {
  std::set<unsigned int> result;
  for (const auto& p : chamber_to_cscRecHit2D_)
    result.insert(p.first);
  return result;
}

std::set<unsigned int> CSCRecHitMatcher::chamberIdsCSCSegment() const {
  std::set<unsigned int> result;
  for (const auto& p : chamber_to_cscSegment_)
    result.insert(p.first);
  return result;
}

const CSCRecHit2DContainer& CSCRecHitMatcher::cscRecHit2DsInLayer(unsigned int detid) const {
  if (layer_to_cscRecHit2D_.find(detid) == layer_to_cscRecHit2D_.end())
    return no_cscRecHit2Ds_;
  return layer_to_cscRecHit2D_.at(detid);
}

const CSCRecHit2DContainer& CSCRecHitMatcher::cscRecHit2DsInChamber(unsigned int detid) const {
  if (chamber_to_cscRecHit2D_.find(detid) == chamber_to_cscRecHit2D_.end())
    return no_cscRecHit2Ds_;
  return chamber_to_cscRecHit2D_.at(detid);
}

const CSCSegmentContainer& CSCRecHitMatcher::cscSegmentsInChamber(unsigned int detid) const {
  if (chamber_to_cscSegment_.find(detid) == chamber_to_cscSegment_.end())
    return no_cscSegments_;
  return chamber_to_cscSegment_.at(detid);
}

int CSCRecHitMatcher::nCSCRecHit2DsInLayer(unsigned int detid) const { return cscRecHit2DsInLayer(detid).size(); }

int CSCRecHitMatcher::nCSCRecHit2DsInChamber(unsigned int detid) const { return cscRecHit2DsInChamber(detid).size(); }

int CSCRecHitMatcher::nCSCSegmentsInChamber(unsigned int detid) const { return cscSegmentsInChamber(detid).size(); }

const CSCRecHit2DContainer CSCRecHitMatcher::cscRecHit2Ds() const {
  CSCRecHit2DContainer result;
  for (const auto& id : chamberIdsCSCRecHit2D()) {
    const auto& segmentsInChamber(cscRecHit2DsInChamber(id));
    result.insert(result.end(), segmentsInChamber.begin(), segmentsInChamber.end());
  }
  return result;
}

const CSCSegmentContainer CSCRecHitMatcher::cscSegments() const {
  CSCSegmentContainer result;
  for (const auto& id : chamberIdsCSCSegment()) {
    const auto& segmentsInChamber(cscSegmentsInChamber(id));
    result.insert(result.end(), segmentsInChamber.begin(), segmentsInChamber.end());
  }
  return result;
}

bool CSCRecHitMatcher::cscRecHit2DInContainer(const CSCRecHit2D& sg, const CSCRecHit2DContainer& c) const {
  bool isSame = false;
  for (const auto& segment : c)
    if (areCSCRecHit2DsSame(sg, segment))
      isSame = true;
  return isSame;
}

bool CSCRecHitMatcher::cscSegmentInContainer(const CSCSegment& sg, const CSCSegmentContainer& c) const {
  bool isSame = false;
  for (const auto& segment : c)
    if (areCSCSegmentsSame(sg, segment))
      isSame = true;
  return isSame;
}

bool CSCRecHitMatcher::isCSCRecHit2DMatched(const CSCRecHit2D& thisSg) const {
  return cscRecHit2DInContainer(thisSg, cscRecHit2Ds());
}

bool CSCRecHitMatcher::isCSCSegmentMatched(const CSCSegment& thisSg) const {
  return cscSegmentInContainer(thisSg, cscSegments());
}

int CSCRecHitMatcher::nCSCRecHit2Ds() const {
  int n = 0;
  const auto& ids = chamberIdsCSCRecHit2D();
  for (const auto& id : ids)
    n += cscRecHit2DsInChamber(id).size();
  return n;
}

int CSCRecHitMatcher::nCSCSegments() const {
  int n = 0;
  const auto& ids = chamberIdsCSCSegment();
  for (const auto& id : ids)
    n += cscSegmentsInChamber(id).size();
  return n;
}

bool CSCRecHitMatcher::areCSCRecHit2DsSame(const CSCRecHit2D& l, const CSCRecHit2D& r) const {
  return l.localPosition() == r.localPosition();
}

bool CSCRecHitMatcher::areCSCSegmentsSame(const CSCSegment& l, const CSCSegment& r) const {
  return (l.localPosition() == r.localPosition() and l.localDirection() == r.localDirection());
}

CSCSegment CSCRecHitMatcher::bestCSCSegment(unsigned int id) {
  CSCSegment emptySegment;
  double chi2overNdf = 99;
  int index = 0;
  int foundIndex = -99;

  for (const auto& seg : chamber_to_cscSegment_[id]) {
    double newChi2overNdf(seg.chi2() / seg.degreesOfFreedom());
    if (newChi2overNdf < chi2overNdf) {
      chi2overNdf = newChi2overNdf;
      foundIndex = index;
    }
    ++index;
  }
  if (foundIndex != -99)
    return chamber_to_cscSegment_[id][foundIndex];
  return emptySegment;
}

GlobalPoint CSCRecHitMatcher::globalPoint(const CSCSegment& c) const {
  return cscGeometry_->idToDet(c.cscDetId())->surface().toGlobal(c.localPosition());
}
