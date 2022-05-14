//
// modified & integrated by Giovanni Abbiendi
// from code by Arun Luthra:
// UserCode/luthra/MuonTrackSelector/src/MuonTrackSelector.cc
//
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimMuon/MCTruth/interface/TrackerMuonHitExtractor.h"
#include <sstream>

TrackerMuonHitExtractor::TrackerMuonHitExtractor(const edm::ParameterSet &parset, edm::ConsumesCollector &&ic)
    : inputDTRecSegment4DToken_(
          ic.consumes<DTRecSegment4DCollection>(parset.getParameter<edm::InputTag>("inputDTRecSegment4DCollection"))),
      inputCSCSegmentToken_(
          ic.consumes<CSCSegmentCollection>(parset.getParameter<edm::InputTag>("inputCSCSegmentCollection"))) {}

void TrackerMuonHitExtractor::init(const edm::Event &iEvent) {
  const edm::Handle<DTRecSegment4DCollection> &dtSegmentCollectionH = iEvent.getHandle(inputDTRecSegment4DToken_);
  const edm::Handle<CSCSegmentCollection> &cscSegmentCollectionH = iEvent.getHandle(inputCSCSegmentToken_);

  edm::LogVerbatim("TrackerMuonHitExtractor") << "\nThere are " << dtSegmentCollectionH->size() << " DT segments.";
  unsigned int index_dt_segment = 0;
  for (DTRecSegment4DCollection::const_iterator segment = dtSegmentCollectionH->begin();
       segment != dtSegmentCollectionH->end();
       ++segment, index_dt_segment++) {
    LocalPoint segmentLocalPosition = segment->localPosition();
    LocalVector segmentLocalDirection = segment->localDirection();
    LocalError segmentLocalPositionError = segment->localPositionError();
    LocalError segmentLocalDirectionError = segment->localDirectionError();
    DetId geoid = segment->geographicalId();
    DTChamberId dtdetid = DTChamberId(geoid);
    int wheel = dtdetid.wheel();
    int station = dtdetid.station();
    int sector = dtdetid.sector();

    float segmentX = segmentLocalPosition.x();
    float segmentY = segmentLocalPosition.y();
    float segmentdXdZ = segmentLocalDirection.x() / segmentLocalDirection.z();
    float segmentdYdZ = segmentLocalDirection.y() / segmentLocalDirection.z();
    float segmentXerr = sqrt(segmentLocalPositionError.xx());
    float segmentYerr = sqrt(segmentLocalPositionError.yy());
    float segmentdXdZerr = sqrt(segmentLocalDirectionError.xx());
    float segmentdYdZerr = sqrt(segmentLocalDirectionError.yy());

    edm::LogVerbatim("TrackerMuonHitExtractor")
        << "\nDT segment index :" << index_dt_segment << "\nchamber Wh:" << wheel << ",St:" << station
        << ",Se:" << sector << "\nLocal Position (X,Y)=(" << segmentX << "," << segmentY << ") +/- (" << segmentXerr
        << "," << segmentYerr << "), "
        << "Local Direction (dXdZ,dYdZ)=(" << segmentdXdZ << "," << segmentdYdZ << ") +/- (" << segmentdXdZerr << ","
        << segmentdYdZerr << ")";
  }

  edm::LogVerbatim("TrackerMuonHitExtractor") << "\nThere are " << cscSegmentCollectionH->size() << " CSC segments.";
  unsigned int index_csc_segment = 0;
  for (CSCSegmentCollection::const_iterator segment = cscSegmentCollectionH->begin();
       segment != cscSegmentCollectionH->end();
       ++segment, index_csc_segment++) {
    LocalPoint segmentLocalPosition = segment->localPosition();
    LocalVector segmentLocalDirection = segment->localDirection();
    LocalError segmentLocalPositionError = segment->localPositionError();
    LocalError segmentLocalDirectionError = segment->localDirectionError();

    DetId geoid = segment->geographicalId();
    CSCDetId cscdetid = CSCDetId(geoid);
    int endcap = cscdetid.endcap();
    int station = cscdetid.station();
    int ring = cscdetid.ring();
    int chamber = cscdetid.chamber();

    float segmentX = segmentLocalPosition.x();
    float segmentY = segmentLocalPosition.y();
    float segmentdXdZ = segmentLocalDirection.x() / segmentLocalDirection.z();
    float segmentdYdZ = segmentLocalDirection.y() / segmentLocalDirection.z();
    float segmentXerr = sqrt(segmentLocalPositionError.xx());
    float segmentYerr = sqrt(segmentLocalPositionError.yy());
    float segmentdXdZerr = sqrt(segmentLocalDirectionError.xx());
    float segmentdYdZerr = sqrt(segmentLocalDirectionError.yy());

    edm::LogVerbatim("TrackerMuonHitExtractor")
        << "\nCSC segment index :" << index_csc_segment << "\nchamber Endcap:" << endcap << ",St:" << station
        << ",Ri:" << ring << ",Ch:" << chamber << "\nLocal Position (X,Y)=(" << segmentX << "," << segmentY << ") +/- ("
        << segmentXerr << "," << segmentYerr << "), "
        << "Local Direction (dXdZ,dYdZ)=(" << segmentdXdZ << "," << segmentdYdZ << ") +/- (" << segmentdXdZerr << ","
        << segmentdYdZerr << ")";
  }
}
std::vector<const TrackingRecHit *> TrackerMuonHitExtractor::getMuonHits(const reco::Muon &mu) const {
  std::vector<const TrackingRecHit *> ret;

  int wheel, station, sector;
  int endcap, /*station, */ ring, chamber;

  edm::LogVerbatim("TrackerMuonHitExtractor")
      << "Number of chambers: " << mu.matches().size()
      << ", arbitrated: " << mu.numberOfMatches(reco::Muon::SegmentAndTrackArbitration)
      << ", tot (no arbitration): " << mu.numberOfMatches(reco::Muon::NoArbitration);
  unsigned int index_chamber = 0;
  int n_segments_noArb = 0;

  for (std::vector<reco::MuonChamberMatch>::const_iterator chamberMatch = mu.matches().begin();
       chamberMatch != mu.matches().end();
       ++chamberMatch, index_chamber++) {
    std::stringstream chamberStr;
    chamberStr << "\nchamber index: " << index_chamber;

    int subdet = chamberMatch->detector();
    DetId did = chamberMatch->id;

    if (subdet == MuonSubdetId::DT) {
      DTChamberId dtdetid = DTChamberId(did);
      wheel = dtdetid.wheel();
      station = dtdetid.station();
      sector = dtdetid.sector();
      chamberStr << ", DT chamber Wh:" << wheel << ",St:" << station << ",Se:" << sector;
    } else if (subdet == MuonSubdetId::CSC) {
      CSCDetId cscdetid = CSCDetId(did);
      endcap = cscdetid.endcap();
      station = cscdetid.station();
      ring = cscdetid.ring();
      chamber = cscdetid.chamber();
      chamberStr << ", CSC chamber End:" << endcap << ",St:" << station << ",Ri:" << ring << ",Ch:" << chamber;
    }

    chamberStr << ", Number of segments: " << chamberMatch->segmentMatches.size();
    edm::LogVerbatim("TrackerMuonHitExtractor") << chamberStr.str();
    n_segments_noArb = n_segments_noArb + chamberMatch->segmentMatches.size();

    unsigned int index_segment = 0;

    for (std::vector<reco::MuonSegmentMatch>::const_iterator segmentMatch = chamberMatch->segmentMatches.begin();
         segmentMatch != chamberMatch->segmentMatches.end();
         ++segmentMatch, index_segment++) {
      float segmentX = segmentMatch->x;
      float segmentY = segmentMatch->y;
      float segmentdXdZ = segmentMatch->dXdZ;
      float segmentdYdZ = segmentMatch->dYdZ;
      float segmentXerr = segmentMatch->xErr;
      float segmentYerr = segmentMatch->yErr;
      float segmentdXdZerr = segmentMatch->dXdZErr;
      float segmentdYdZerr = segmentMatch->dYdZErr;

      CSCSegmentRef segmentCSC = segmentMatch->cscSegmentRef;
      DTRecSegment4DRef segmentDT = segmentMatch->dtSegmentRef;

      bool segment_arbitrated_Ok = (segmentMatch->isMask(reco::MuonSegmentMatch::BestInChamberByDR) &&
                                    segmentMatch->isMask(reco::MuonSegmentMatch::BelongsToTrackByDR));

      std::string ARBITRATED(" ***Arbitrated Off*** ");
      if (segment_arbitrated_Ok)
        ARBITRATED = " ***ARBITRATED OK*** ";

      if (subdet == MuonSubdetId::DT) {
        edm::LogVerbatim("TrackerMuonHitExtractor")
            << "\n\t segment index: " << index_segment << ARBITRATED << "\n\t  Local Position (X,Y)=(" << segmentX
            << "," << segmentY << ") +/- (" << segmentXerr << "," << segmentYerr << "), "
            << "\n\t  Local Direction (dXdZ,dYdZ)=(" << segmentdXdZ << "," << segmentdYdZ << ") +/- (" << segmentdXdZerr
            << "," << segmentdYdZerr << ")";

        // with the following line the DT segments failing standard arbitration
        // are skipped
        if (!segment_arbitrated_Ok)
          continue;

        if (segmentDT.get() != nullptr) {
          const DTRecSegment4D *segment = segmentDT.get();

          edm::LogVerbatim("TrackerMuonHitExtractor")
              << "\t ===> MATCHING with DT segment with index = " << segmentDT.key();

          if (segment->hasPhi()) {
            const DTChamberRecSegment2D *phiSeg = segment->phiSegment();
            std::vector<const TrackingRecHit *> phiHits = phiSeg->recHits();
            for (std::vector<const TrackingRecHit *>::const_iterator ihit = phiHits.begin(); ihit != phiHits.end();
                 ++ihit) {
              ret.push_back(*ihit);
            }
          }

          if (segment->hasZed()) {
            const DTSLRecSegment2D *zSeg = (*segment).zSegment();
            std::vector<const TrackingRecHit *> zedHits = zSeg->recHits();
            for (std::vector<const TrackingRecHit *>::const_iterator ihit = zedHits.begin(); ihit != zedHits.end();
                 ++ihit) {
              ret.push_back(*ihit);
            }
          }
        } else
          edm::LogWarning("TrackerMuonHitExtractor") << "\n***WARNING: UNMATCHED DT segment ! \n";
      }  // if (subdet == MuonSubdetId::DT)

      else if (subdet == MuonSubdetId::CSC) {
        edm::LogVerbatim("TrackerMuonHitExtractor")
            << "\n\t segment index: " << index_segment << ARBITRATED << "\n\t  Local Position (X,Y)=(" << segmentX
            << "," << segmentY << ") +/- (" << segmentXerr << "," << segmentYerr << "), "
            << "\n\t  Local Direction (dXdZ,dYdZ)=(" << segmentdXdZ << "," << segmentdYdZ << ") +/- (" << segmentdXdZerr
            << "," << segmentdYdZerr << ")";

        // with the following line the CSC segments failing standard arbitration
        // are skipped
        if (!segment_arbitrated_Ok)
          continue;

        if (segmentCSC.get() != nullptr) {
          const CSCSegment *segment = segmentCSC.get();

          edm::LogVerbatim("TrackerMuonHitExtractor")
              << "\t ===> MATCHING with CSC segment with index = " << segmentCSC.key();

          std::vector<const TrackingRecHit *> hits = segment->recHits();
          for (std::vector<const TrackingRecHit *>::const_iterator ihit = hits.begin(); ihit != hits.end(); ++ihit) {
            ret.push_back(*ihit);
          }
        } else
          edm::LogWarning("TrackerMuonHitExtractor") << "\n***WARNING: UNMATCHED CSC segment ! \n";
      }  // else if (subdet == MuonSubdetId::CSC)

    }  // loop on vector<MuonSegmentMatch>
  }    // loop on vector<MuonChamberMatch>

  edm::LogVerbatim("TrackerMuonHitExtractor") << "\n N. matched Segments before arbitration = " << n_segments_noArb;

  return ret;
}
