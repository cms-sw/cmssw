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
#include "SimMuon/MCTruth/plugins/MuonTrackProducer.h"
#include <sstream>

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

MuonTrackProducer::MuonTrackProducer(const edm::ParameterSet &parset)
    : muonsToken(consumes<reco::MuonCollection>(parset.getParameter<edm::InputTag>("muonsTag"))),
      inputDTRecSegment4DToken_(
          consumes<DTRecSegment4DCollection>(parset.getParameter<edm::InputTag>("inputDTRecSegment4DCollection"))),
      inputCSCSegmentToken_(
          consumes<CSCSegmentCollection>(parset.getParameter<edm::InputTag>("inputCSCSegmentCollection"))),
      selectionTags(parset.getParameter<std::vector<std::string>>("selectionTags")),
      trackType(parset.getParameter<std::string>("trackType")),
      ignoreMissingMuonCollection(parset.getUntrackedParameter<bool>("ignoreMissingMuonCollection", false)),
      parset_(parset) {
  edm::LogVerbatim("MuonTrackProducer") << "constructing  MuonTrackProducer" << parset_.dump();
  produces<reco::TrackCollection>();
  produces<reco::TrackExtraCollection>();
  produces<TrackingRecHitCollection>();
}

MuonTrackProducer::~MuonTrackProducer() {}

void MuonTrackProducer::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  bool muonAvailable = iEvent.getByToken(muonsToken, muonCollectionH);
  if (ignoreMissingMuonCollection && !muonAvailable)
    edm::LogVerbatim("MuonTrackProducer") << "\n ignoring missing muon collection.";

  else {
    iEvent.getByToken(inputDTRecSegment4DToken_, dtSegmentCollectionH_);
    iEvent.getByToken(inputCSCSegmentToken_, cscSegmentCollectionH_);

    edm::ESHandle<TrackerTopology> httopo;
    iSetup.get<TrackerTopologyRcd>().get(httopo);
    const TrackerTopology &ttopo = *httopo;

    std::unique_ptr<reco::TrackCollection> selectedTracks(new reco::TrackCollection);
    std::unique_ptr<reco::TrackExtraCollection> selectedTrackExtras(new reco::TrackExtraCollection());
    std::unique_ptr<TrackingRecHitCollection> selectedTrackHits(new TrackingRecHitCollection());

    reco::TrackRefProd rTracks = iEvent.getRefBeforePut<reco::TrackCollection>();
    reco::TrackExtraRefProd rTrackExtras = iEvent.getRefBeforePut<reco::TrackExtraCollection>();
    TrackingRecHitRefProd rHits = iEvent.getRefBeforePut<TrackingRecHitCollection>();

    edm::Ref<reco::TrackExtraCollection>::key_type idx = 0;
    edm::Ref<reco::TrackExtraCollection>::key_type hidx = 0;

    edm::LogVerbatim("MuonTrackProducer") << "\nThere are " << dtSegmentCollectionH_->size() << " DT segments.";
    unsigned int index_dt_segment = 0;
    for (DTRecSegment4DCollection::const_iterator segment = dtSegmentCollectionH_->begin();
         segment != dtSegmentCollectionH_->end();
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

      edm::LogVerbatim("MuonTrackProducer")
          << "\nDT segment index :" << index_dt_segment << "\nchamber Wh:" << wheel << ",St:" << station
          << ",Se:" << sector << "\nLocal Position (X,Y)=(" << segmentX << "," << segmentY << ") +/- (" << segmentXerr
          << "," << segmentYerr << "), "
          << "Local Direction (dXdZ,dYdZ)=(" << segmentdXdZ << "," << segmentdYdZ << ") +/- (" << segmentdXdZerr << ","
          << segmentdYdZerr << ")";
    }

    edm::LogVerbatim("MuonTrackProducer") << "\nThere are " << cscSegmentCollectionH_->size() << " CSC segments.";
    unsigned int index_csc_segment = 0;
    for (CSCSegmentCollection::const_iterator segment = cscSegmentCollectionH_->begin();
         segment != cscSegmentCollectionH_->end();
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

      edm::LogVerbatim("MuonTrackProducer")
          << "\nCSC segment index :" << index_csc_segment << "\nchamber Endcap:" << endcap << ",St:" << station
          << ",Ri:" << ring << ",Ch:" << chamber << "\nLocal Position (X,Y)=(" << segmentX << "," << segmentY
          << ") +/- (" << segmentXerr << "," << segmentYerr << "), "
          << "Local Direction (dXdZ,dYdZ)=(" << segmentdXdZ << "," << segmentdYdZ << ") +/- (" << segmentdXdZerr << ","
          << segmentdYdZerr << ")";
    }

    edm::LogVerbatim("MuonTrackProducer") << "\nThere are " << muonCollectionH->size() << " reco::Muons.";
    unsigned int muon_index = 0;
    for (reco::MuonCollection::const_iterator muon = muonCollectionH->begin(); muon != muonCollectionH->end();
         ++muon, muon_index++) {
      edm::LogVerbatim("MuonTrackProducer") << "\n******* muon index : " << muon_index;

      std::vector<bool> isGood;
      for (unsigned int index = 0; index < selectionTags.size(); ++index) {
        isGood.push_back(false);

        muon::SelectionType muonType = muon::selectionTypeFromString(selectionTags[index]);
        isGood[index] = muon::isGoodMuon(*muon, muonType);
      }

      bool isGoodResult = true;
      for (unsigned int index = 0; index < isGood.size(); ++index) {
        isGoodResult *= isGood[index];
      }

      if (isGoodResult) {
        // new copy of Track
        reco::TrackRef trackref;
        bool addMatchedMuonSegments = false;

        if (trackType == "innerTrack") {
          if (muon->innerTrack().isNonnull())
            trackref = muon->innerTrack();
          else
            continue;
        } else if (trackType == "outerTrack") {
          if (muon->outerTrack().isNonnull())
            trackref = muon->outerTrack();
          else
            continue;
        } else if (trackType == "globalTrack") {
          if (muon->globalTrack().isNonnull())
            trackref = muon->globalTrack();
          else
            continue;
        } else if (trackType == "innerTrackPlusSegments") {
          if (muon->innerTrack().isNonnull()) {
            trackref = muon->innerTrack();
            addMatchedMuonSegments = true;
          } else
            continue;
        } else if (trackType == "rpcMuonTrack") {
          if (muon->innerTrack().isNonnull() && muon->isRPCMuon()) {
            trackref = muon->innerTrack();
          } else
            continue;
        } else if (trackType == "gemMuonTrack") {
          if (muon->innerTrack().isNonnull() && muon->isGEMMuon()) {
            trackref = muon->innerTrack();
          } else
            continue;
        } else if (trackType == "me0MuonTrack") {
          if (muon->innerTrack().isNonnull() && muon->isME0Muon()) {
            trackref = muon->innerTrack();
          } else
            continue;
        } else if (trackType == "tunepTrack") {
          if (muon->isGlobalMuon() && muon->tunePMuonBestTrack().isNonnull())
            trackref = muon->tunePMuonBestTrack();
          else
            continue;
        } else if (trackType == "pfTrack") {
          if (muon->isPFMuon() && muon->muonBestTrack().isNonnull())
            trackref = muon->muonBestTrack();
          else
            continue;
        } else if (trackType == "recomuonTrack") {
          if (muon->isGlobalMuon())
            trackref = muon->globalTrack();
          else if (muon->isTrackerMuon()) {
            trackref = muon->innerTrack();
            addMatchedMuonSegments = true;
          } else if (muon->isStandAloneMuon())
            trackref = muon->outerTrack();
          else if (muon->isRPCMuon())
            trackref = muon->innerTrack();
          else if (muon->isGEMMuon())
            trackref = muon->innerTrack();
          else if (muon->isME0Muon())
            trackref = muon->innerTrack();
          else
            trackref = muon->muonBestTrack();

          if (muon->muonBestTrackType() != muon->tunePMuonBestTrackType())
            edm::LogVerbatim("MuonTrackProducer") << "\n *** PF != TuneP *** \n" << std::endl;

          edm::LogVerbatim("MuonTrackProducer") << "isGlobal     ? " << muon->isGlobalMuon() << std::endl;
          edm::LogVerbatim("MuonTrackProducer")
              << "isTracker    ? " << muon->isTrackerMuon() << ", isRPC ? " << muon->isRPCMuon() << ", isGEM ? "
              << muon->isGEMMuon() << ", isME0 ? " << muon->isME0Muon() << std::endl;
          edm::LogVerbatim("MuonTrackProducer") << "isStandAlone ? " << muon->isStandAloneMuon() << std::endl;
          edm::LogVerbatim("MuonTrackProducer") << "isCalo ? " << muon->isCaloMuon() << std::endl;
          edm::LogVerbatim("MuonTrackProducer") << "isPF         ? " << muon->isPFMuon() << std::endl << std::endl;

          edm::LogVerbatim("MuonTrackProducer")
              << " enum MuonTrackType {None, InnerTrack, OuterTrack, CombinedTrack, TPFMS, Picky, DYT }" << std::endl;

          edm::LogVerbatim("MuonTrackProducer")
              << "(muon) pt =   " << muon->pt() << ", eta = " << muon->eta() << ", phi = " << muon->phi() << std::endl;

          if (muon->muonBestTrack().isNonnull())
            edm::LogVerbatim("MuonTrackProducer")
                << "(best) pt =   " << muon->muonBestTrack()->pt() << ", eta = " << muon->muonBestTrack()->eta()
                << ", phi = " << muon->muonBestTrack()->phi()
                << ", N mu hits = " << muon->muonBestTrack()->hitPattern().numberOfValidMuonHits()
                << ", N trk hits = " << muon->muonBestTrack()->hitPattern().numberOfValidTrackerHits()
                << ", MuonTrackType = " << muon->muonBestTrackType() << std::endl;
          if (muon->tunePMuonBestTrack().isNonnull())
            edm::LogVerbatim("MuonTrackProducer")
                << "(tuneP) pt =  " << muon->tunePMuonBestTrack()->pt()
                << ", eta = " << muon->tunePMuonBestTrack()->eta() << ", phi = " << muon->tunePMuonBestTrack()->phi()
                << ", N mu hits = " << muon->tunePMuonBestTrack()->hitPattern().numberOfValidMuonHits()
                << ", N trk hits = " << muon->tunePMuonBestTrack()->hitPattern().numberOfValidTrackerHits()
                << ", MuonTrackType = " << muon->tunePMuonBestTrackType() << std::endl;
          if (muon->innerTrack().isNonnull())
            edm::LogVerbatim("MuonTrackProducer")
                << "(inner) pt =  " << muon->innerTrack()->pt() << ", eta = " << muon->innerTrack()->eta()
                << ", phi = " << muon->innerTrack()->phi()
                << ", N trk hits = " << muon->innerTrack()->hitPattern().numberOfValidTrackerHits() << std::endl;
          if (muon->globalTrack().isNonnull())
            edm::LogVerbatim("MuonTrackProducer")
                << "(global) pt = " << muon->globalTrack()->pt() << ", eta = " << muon->globalTrack()->eta()
                << ", phi = " << muon->globalTrack()->phi()
                << ", N mu hits = " << muon->globalTrack()->hitPattern().numberOfValidMuonHits()
                << ", N trk hits = " << muon->globalTrack()->hitPattern().numberOfValidTrackerHits() << std::endl;
          if (muon->outerTrack().isNonnull())
            edm::LogVerbatim("MuonTrackProducer")
                << "(outer) pt =  " << muon->outerTrack()->pt() << ", eta = " << muon->outerTrack()->eta()
                << ", phi = " << muon->outerTrack()->phi()
                << ", N mu hits = " << muon->outerTrack()->hitPattern().numberOfValidMuonHits() << std::endl;
          if (muon->tpfmsTrack().isNonnull())
            edm::LogVerbatim("MuonTrackProducer")
                << "(tpfms) pt =  " << muon->tpfmsTrack()->pt() << ", eta = " << muon->tpfmsTrack()->eta()
                << ", phi = " << muon->tpfmsTrack()->phi()
                << ", N mu hits = " << muon->tpfmsTrack()->hitPattern().numberOfValidMuonHits()
                << ", N trk hits = " << muon->tpfmsTrack()->hitPattern().numberOfValidTrackerHits() << std::endl;
          if (muon->pickyTrack().isNonnull())
            edm::LogVerbatim("MuonTrackProducer")
                << "(picky) pt =  " << muon->pickyTrack()->pt() << ", eta = " << muon->pickyTrack()->eta()
                << ", phi = " << muon->pickyTrack()->phi()
                << ", N mu hits = " << muon->pickyTrack()->hitPattern().numberOfValidMuonHits()
                << ", N trk hits = " << muon->pickyTrack()->hitPattern().numberOfValidTrackerHits() << std::endl;
          if (muon->dytTrack().isNonnull())
            edm::LogVerbatim("MuonTrackProducer")
                << "(dyt) pt =    " << muon->dytTrack()->pt() << ", eta = " << muon->dytTrack()->eta()
                << ", phi = " << muon->dytTrack()->phi()
                << ", N mu hits = " << muon->dytTrack()->hitPattern().numberOfValidMuonHits()
                << ", N trk hits = " << muon->dytTrack()->hitPattern().numberOfValidTrackerHits() << std::endl;
        }

        edm::LogVerbatim("MuonTrackProducer") << "\t *** Selected *** ";
        const reco::Track *trk = &(*trackref);
        // pointer to old track:
        std::unique_ptr<reco::Track> newTrk(new reco::Track(*trk));

        newTrk->setExtra(reco::TrackExtraRef(rTrackExtras, idx++));
        PropagationDirection seedDir = trk->seedDirection();
        // new copy of track Extras
        std::unique_ptr<reco::TrackExtra> newExtra(new reco::TrackExtra(trk->outerPosition(),
                                                                        trk->outerMomentum(),
                                                                        trk->outerOk(),
                                                                        trk->innerPosition(),
                                                                        trk->innerMomentum(),
                                                                        trk->innerOk(),
                                                                        trk->outerStateCovariance(),
                                                                        trk->outerDetId(),
                                                                        trk->innerStateCovariance(),
                                                                        trk->innerDetId(),
                                                                        seedDir));

        // new copy of the silicon hits; add hit refs to Extra and hits to hit
        // collection

        //      edm::LogVerbatim("MuonTrackProducer")<<"\n printing initial
        //      hit_pattern"; trk->hitPattern().print();
        unsigned int nHitsToAdd = 0;
        for (trackingRecHit_iterator iHit = trk->recHitsBegin(); iHit != trk->recHitsEnd(); iHit++) {
          TrackingRecHit *hit = (*iHit)->clone();
          selectedTrackHits->push_back(hit);
          ++nHitsToAdd;
        }

        if (addMatchedMuonSegments) {
          int wheel, station, sector;
          int endcap, /*station, */ ring, chamber;

          edm::LogVerbatim("MuonTrackProducer")
              << "Number of chambers: " << muon->matches().size()
              << ", arbitrated: " << muon->numberOfMatches(reco::Muon::SegmentAndTrackArbitration);
          unsigned int index_chamber = 0;

          for (std::vector<reco::MuonChamberMatch>::const_iterator chamberMatch = muon->matches().begin();
               chamberMatch != muon->matches().end();
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
            edm::LogVerbatim("MuonTrackProducer") << chamberStr.str();

            unsigned int index_segment = 0;

            for (std::vector<reco::MuonSegmentMatch>::const_iterator segmentMatch =
                     chamberMatch->segmentMatches.begin();
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
                edm::LogVerbatim("MuonTrackProducer")
                    << "\n\t segment index: " << index_segment << ARBITRATED << "\n\t  Local Position (X,Y)=("
                    << segmentX << "," << segmentY << ") +/- (" << segmentXerr << "," << segmentYerr << "), "
                    << "\n\t  Local Direction (dXdZ,dYdZ)=(" << segmentdXdZ << "," << segmentdYdZ << ") +/- ("
                    << segmentdXdZerr << "," << segmentdYdZerr << ")";

                if (!segment_arbitrated_Ok)
                  continue;

                if (segmentDT.get() != nullptr) {
                  const DTRecSegment4D *segment = segmentDT.get();

                  edm::LogVerbatim("MuonTrackProducer")
                      << "\t ===> MATCHING with DT segment with index = " << segmentDT.key();

                  if (segment->hasPhi()) {
                    const DTChamberRecSegment2D *phiSeg = segment->phiSegment();
                    std::vector<const TrackingRecHit *> phiHits = phiSeg->recHits();
                    for (std::vector<const TrackingRecHit *>::const_iterator ihit = phiHits.begin();
                         ihit != phiHits.end();
                         ++ihit) {
                      TrackingRecHit *seghit = (*ihit)->clone();
                      newTrk->appendHitPattern(*seghit, ttopo);
                      //		    edm::LogVerbatim("MuonTrackProducer")<<"hit
                      // pattern for position "<<index_hit<<" set to:";
                      //		    newTrk->hitPattern().printHitPattern(index_hit,
                      // std::cout);
                      selectedTrackHits->push_back(seghit);
                      ++nHitsToAdd;
                    }
                  }

                  if (segment->hasZed()) {
                    const DTSLRecSegment2D *zSeg = (*segment).zSegment();
                    std::vector<const TrackingRecHit *> zedHits = zSeg->recHits();
                    for (std::vector<const TrackingRecHit *>::const_iterator ihit = zedHits.begin();
                         ihit != zedHits.end();
                         ++ihit) {
                      TrackingRecHit *seghit = (*ihit)->clone();
                      newTrk->appendHitPattern(*seghit, ttopo);
                      //		    edm::LogVerbatim("MuonTrackProducer")<<"hit
                      // pattern for position "<<index_hit<<" set to:";
                      //		    newTrk->hitPattern().printHitPattern(index_hit,
                      // std::cout);
                      selectedTrackHits->push_back(seghit);
                      ++nHitsToAdd;
                    }
                  }
                } else
                  edm::LogWarning("MuonTrackProducer") << "\n***WARNING: UNMATCHED DT segment ! \n";
              }  // if (subdet == MuonSubdetId::DT)

              else if (subdet == MuonSubdetId::CSC) {
                edm::LogVerbatim("MuonTrackProducer")
                    << "\n\t segment index: " << index_segment << ARBITRATED << "\n\t  Local Position (X,Y)=("
                    << segmentX << "," << segmentY << ") +/- (" << segmentXerr << "," << segmentYerr << "), "
                    << "\n\t  Local Direction (dXdZ,dYdZ)=(" << segmentdXdZ << "," << segmentdYdZ << ") +/- ("
                    << segmentdXdZerr << "," << segmentdYdZerr << ")";

                if (!segment_arbitrated_Ok)
                  continue;

                if (segmentCSC.get() != nullptr) {
                  const CSCSegment *segment = segmentCSC.get();

                  edm::LogVerbatim("MuonTrackProducer")
                      << "\t ===> MATCHING with CSC segment with index = " << segmentCSC.key();

                  std::vector<const TrackingRecHit *> hits = segment->recHits();
                  for (std::vector<const TrackingRecHit *>::const_iterator ihit = hits.begin(); ihit != hits.end();
                       ++ihit) {
                    TrackingRecHit *seghit = (*ihit)->clone();
                    newTrk->appendHitPattern(*seghit, ttopo);
                    //		    edm::LogVerbatim("MuonTrackProducer")<<"hit
                    // pattern for position "<<index_hit<<" set to:";
                    //		    newTrk->hitPattern().printHitPattern(index_hit,
                    // std::cout);
                    selectedTrackHits->push_back(seghit);
                    ++nHitsToAdd;
                  }
                } else
                  edm::LogWarning("MuonTrackProducer") << "\n***WARNING: UNMATCHED CSC segment ! \n";
              }  //  else if (subdet == MuonSubdetId::CSC)

            }  // loop on vector<MuonSegmentMatch>
          }    // loop on vector<MuonChamberMatch>
        }      // if (trackType == "innerTrackPlusSegments")

        //      edm::LogVerbatim("MuonTrackProducer")<<"\n printing final
        //      hit_pattern"; newTrk->hitPattern().print();

        newExtra->setHits(rHits, hidx, nHitsToAdd);
        hidx += nHitsToAdd;

        selectedTracks->push_back(*newTrk);
        selectedTrackExtras->push_back(*newExtra);

      }  // if (isGoodResult)
    }    // loop on reco::MuonCollection

    iEvent.put(std::move(selectedTracks));
    iEvent.put(std::move(selectedTrackExtras));
    iEvent.put(std::move(selectedTrackHits));
  }
}
