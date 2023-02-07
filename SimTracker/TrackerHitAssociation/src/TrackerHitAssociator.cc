// File: TrackerHitAssociator.cc

#include <memory>
#include <string>
#include <vector>

#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//--- for Geometry:
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"

//for accumulate
#include <numeric>
#include <iostream>

using namespace std;
using namespace edm;

//
// Constructor for Config helper class, using default parameters
//
TrackerHitAssociator::Config::Config(edm::ConsumesCollector&& iC)
    : doPixel_(true), doStrip_(true), useOTph2_(false), doTrackAssoc_(false), assocHitbySimTrack_(false) {
  if (doStrip_) {
    if (useOTph2_)
      ph2OTrToken_ = iC.consumes<edm::DetSetVector<PixelDigiSimLink>>(edm::InputTag("simSiPixelDigis", "Tracker"));
    else
      stripToken_ = iC.consumes<edm::DetSetVector<StripDigiSimLink>>(edm::InputTag("simSiStripDigis"));
  }
  if (doPixel_) {
    if (useOTph2_)
      pixelToken_ = iC.consumes<edm::DetSetVector<PixelDigiSimLink>>(edm::InputTag("simSiPixelDigis", "Pixel"));
    else
      pixelToken_ = iC.consumes<edm::DetSetVector<PixelDigiSimLink>>(edm::InputTag("simSiPixelDigis"));
  }
  if (!doTrackAssoc_) {
    std::vector<std::string> trackerContainers;
    trackerContainers.reserve(12);
    trackerContainers.emplace_back("g4SimHitsTrackerHitsTIBLowTof");
    trackerContainers.emplace_back("g4SimHitsTrackerHitsTIBHighTof");
    trackerContainers.emplace_back("g4SimHitsTrackerHitsTIDLowTof");
    trackerContainers.emplace_back("g4SimHitsTrackerHitsTIDHighTof");
    trackerContainers.emplace_back("g4SimHitsTrackerHitsTOBLowTof");
    trackerContainers.emplace_back("g4SimHitsTrackerHitsTOBHighTof");
    trackerContainers.emplace_back("g4SimHitsTrackerHitsTECLowTof");
    trackerContainers.emplace_back("g4SimHitsTrackerHitsTECHighTof");
    trackerContainers.emplace_back("g4SimHitsTrackerHitsPixelBarrelLowTof");
    trackerContainers.emplace_back("g4SimHitsTrackerHitsPixelBarrelHighTof");
    trackerContainers.emplace_back("g4SimHitsTrackerHitsPixelEndcapLowTof");
    trackerContainers.emplace_back("g4SimHitsTrackerHitsPixelEndcapHighTof");
    cfTokens_.reserve(trackerContainers.size());
    simHitTokens_.reserve(trackerContainers.size());
    for (auto const& trackerContainer : trackerContainers) {
      cfTokens_.push_back(iC.consumes<CrossingFrame<PSimHit>>(edm::InputTag("mix", trackerContainer)));
      simHitTokens_.push_back(iC.consumes<std::vector<PSimHit>>(edm::InputTag("g4SimHits", trackerContainer)));
    }
  }
}

//
// Constructor for Config helper class, using configured parameters
//
TrackerHitAssociator::Config::Config(const edm::ParameterSet& conf, edm::ConsumesCollector&& iC)
    : doPixel_(conf.getParameter<bool>("associatePixel")),
      doStrip_(conf.getParameter<bool>("associateStrip")),
      useOTph2_(conf.existsAs<bool>("usePhase2Tracker") ? conf.getParameter<bool>("usePhase2Tracker") : false),
      //
      doTrackAssoc_(conf.getParameter<bool>("associateRecoTracks")),
      assocHitbySimTrack_(
          conf.existsAs<bool>("associateHitbySimTrack") ? conf.getParameter<bool>("associateHitbySimTrack") : false) {
  if (doStrip_) {
    if (useOTph2_)
      ph2OTrToken_ =
          iC.consumes<edm::DetSetVector<PixelDigiSimLink>>(conf.getParameter<edm::InputTag>("phase2TrackerSimLinkSrc"));
    else
      stripToken_ =
          iC.consumes<edm::DetSetVector<StripDigiSimLink>>(conf.getParameter<edm::InputTag>("stripSimLinkSrc"));
  }
  if (doPixel_)
    pixelToken_ = iC.consumes<edm::DetSetVector<PixelDigiSimLink>>(conf.getParameter<edm::InputTag>("pixelSimLinkSrc"));
  if (!doTrackAssoc_) {
    std::vector<std::string> trackerContainers(conf.getParameter<std::vector<std::string>>("ROUList"));
    cfTokens_.reserve(trackerContainers.size());
    simHitTokens_.reserve(trackerContainers.size());
    for (auto const& trackerContainer : trackerContainers) {
      cfTokens_.push_back(iC.consumes<CrossingFrame<PSimHit>>(edm::InputTag("mix", trackerContainer)));
      simHitTokens_.push_back(iC.consumes<std::vector<PSimHit>>(edm::InputTag("g4SimHits", trackerContainer)));
    }
  }
}

void TrackerHitAssociator::fillPSetDescription(edm::ParameterSetDescription& desc) {
  desc.setComment("auxilliary class to store information about recHit/simHit association");
  desc.add<bool>("associatePixel", false);
  desc.add<bool>("associateStrip", false);
  desc.add<bool>("usePhase2Tracker", false);
  desc.add<bool>("associateRecoTracks", false);
  desc.add<bool>("associateHitbySimTrack", false);
  desc.add<edm::InputTag>("phase2TrackerSimLinkSrc", edm::InputTag("simSiPixelDigis", "Tracker"));
  desc.add<edm::InputTag>("stripSimLinkSrc", edm::InputTag("simSiStripDigis"));
  desc.add<edm::InputTag>("pixelSimLinkSrc", edm::InputTag("simSiPixelDigis"));
  desc.add<std::vector<std::string>>(
      "ROUList", {"TrackerHitsTIBLowTof", "TrackerHitsTIBHighTof", "TrackerHitsTOBLowTof", "TrackerHitsTOBHighTof"});
}

//
// Constructor supporting consumes interface
//
TrackerHitAssociator::TrackerHitAssociator(const edm::Event& e, const TrackerHitAssociator::Config& config)
    : doPixel_(config.doPixel_),
      doStrip_(config.doStrip_),
      useOTph2_(config.useOTph2_),
      doTrackAssoc_(config.doTrackAssoc_),
      assocHitbySimTrack_(config.assocHitbySimTrack_) {
  //if track association there is no need to access the input collections
  if (!doTrackAssoc_) {
    makeMaps(e, config);
  }

  if (doStrip_) {
    if (useOTph2_)
      e.getByToken(config.ph2OTrToken_, ph2trackerdigisimlink);
    else
      e.getByToken(config.stripToken_, stripdigisimlink);
  }
  if (doPixel_)
    e.getByToken(config.pixelToken_, pixeldigisimlink);
}

void TrackerHitAssociator::makeMaps(const edm::Event& theEvent, const TrackerHitAssociator::Config& config) {
  // Step A: Get Inputs
  //  The collections are specified via ROUList in the configuration, and can
  //  be either crossing frames (e.g., mix/g4SimHitsTrackerHitsTIBLowTof)
  //  or just PSimHits (e.g., g4SimHits/TrackerHitsTIBLowTof)
  if (assocHitbySimTrack_) {
    for (auto const& cfToken : config.cfTokens_) {
      edm::Handle<CrossingFrame<PSimHit>> cf_simhit;
      int Nhits = 0;
      if (theEvent.getByToken(cfToken, cf_simhit)) {
        std::unique_ptr<MixCollection<PSimHit>> thisContainerHits(new MixCollection<PSimHit>(cf_simhit.product()));
        for (auto const& isim : *thisContainerHits) {
          DetId theDet(isim.detUnitId());
          SimHitMap[theDet].push_back(isim);
          ++Nhits;
        }
        LogDebug("TrkHitAssocTrace") << "simHits from crossing frames; map size = " << SimHitMap.size()
                                     << ", Hit count = " << Nhits << std::endl;
      }
    }
    for (auto const& simHitToken : config.simHitTokens_) {
      edm::Handle<std::vector<PSimHit>> simHits;
      int Nhits = 0;
      if (theEvent.getByToken(simHitToken, simHits)) {
        for (auto const& isim : *simHits) {
          DetId theDet(isim.detUnitId());
          SimHitMap[theDet].push_back(isim);
          ++Nhits;
        }
        LogDebug("TrkHitAssocTrace") << "simHits from prompt collections; map size = " << SimHitMap.size()
                                     << ", Hit count = " << Nhits << std::endl;
      }
    }
  } else {  // !assocHitbySimTrack_
    const char* const highTag = "HighTof";
    unsigned int tofBin;
    edm::EDConsumerBase::Labels labels;
    subDetTofBin theSubDetTofBin;
    unsigned int collectionIndex = 0;
    for (auto const& cfToken : config.cfTokens_) {
      collectionIndex++;
      edm::Handle<CrossingFrame<PSimHit>> cf_simhit;
      int Nhits = 0;
      if (theEvent.getByToken(cfToken, cf_simhit)) {
        std::unique_ptr<MixCollection<PSimHit>> thisContainerHits(new MixCollection<PSimHit>(cf_simhit.product()));
        theEvent.labelsForToken(cfToken, labels);
        if (std::strstr(labels.productInstance, highTag) != nullptr) {
          tofBin = StripDigiSimLink::HighTof;
        } else {
          tofBin = StripDigiSimLink::LowTof;
        }
        for (auto const& isim : *thisContainerHits) {
          DetId theDet(isim.detUnitId());
          theSubDetTofBin = std::make_pair(theDet.subdetId(), tofBin);
          SimHitCollMap[theSubDetTofBin] = collectionIndex;
          SimHitMap[SimHitCollMap[theSubDetTofBin]].push_back(isim);
          ++Nhits;
        }
        LogDebug("TrkHitAssocTrace") << "simHits from crossing frames " << collectionIndex << ":  " << Nhits
                                     << std::endl;
      }
    }
    collectionIndex = 0;
    for (auto const& simHitToken : config.simHitTokens_) {
      collectionIndex++;
      edm::Handle<std::vector<PSimHit>> simHits;
      int Nhits = 0;
      if (theEvent.getByToken(simHitToken, simHits)) {
        theEvent.labelsForToken(simHitToken, labels);
        if (std::strstr(labels.productInstance, highTag) != nullptr) {
          tofBin = StripDigiSimLink::HighTof;
        } else {
          tofBin = StripDigiSimLink::LowTof;
        }
        for (auto const& isim : *simHits) {
          DetId theDet(isim.detUnitId());
          theSubDetTofBin = std::make_pair(theDet.subdetId(), tofBin);
          SimHitCollMap[theSubDetTofBin] = collectionIndex;
          SimHitMap[SimHitCollMap[theSubDetTofBin]].push_back(isim);
          ++Nhits;
        }
        LogDebug("TrkHitAssocTrace") << "simHits from prompt collection " << collectionIndex << ":  " << Nhits
                                     << std::endl;
      }
    }
  }
}

std::vector<PSimHit> TrackerHitAssociator::associateHit(const TrackingRecHit& thit) const {
  if (const SiTrackerMultiRecHit* rechit = dynamic_cast<const SiTrackerMultiRecHit*>(&thit)) {
    return associateMultiRecHit(rechit);
  }

  //vector with the matched SimHit
  std::vector<PSimHit> result;

  if (doTrackAssoc_)
    return result;  // We don't want the SimHits for this RecHit

  // Vectors to contain lists of matched simTracks, simHits
  std::vector<SimHitIdpr> simtrackid;
  std::vector<simhitAddr> simhitCFPos;

  //get the Detector type of the rechit
  DetId detid = thit.geographicalId();
  uint32_t detID = detid.rawId();

  // Get the vectors of simtrackIDs and simHit addresses associated with this rechit
  associateHitId(thit, simtrackid, &simhitCFPos);
  LogDebug("TrkHitAssocTrace") << printDetBnchEvtTrk(detid, detID, simtrackid);

  // Get the vector of simHits associated with this rechit
  if (!assocHitbySimTrack_ && !simhitCFPos.empty()) {
    // We use the indices to the simHit collections taken
    //  from the DigiSimLinks and returned in simhitCFPos.
    //  simhitCFPos[i] contains the full address of the ith simhit:
    //   <collection index, simhit index>

    //check if the recHit is a SiStripMatchedRecHit2D
    if (dynamic_cast<const SiStripMatchedRecHit2D*>(&thit)) {
      for (auto const& theSimHitAddr : simhitCFPos) {
        simHitCollectionID theSimHitCollID = theSimHitAddr.first;
        auto it = SimHitMap.find(theSimHitCollID);
        if (it != SimHitMap.end()) {
          unsigned int theSimHitIndex = theSimHitAddr.second;
          if (theSimHitIndex < (it->second).size()) {
            const PSimHit& theSimHit = (it->second)[theSimHitIndex];
            // Try to remove ghosts by requiring a match to the simTrack also
            unsigned int simHitid = theSimHit.trackId();
            EncodedEventId simHiteid = theSimHit.eventId();
            for (auto const& id : simtrackid) {
              if (simHitid == id.first && simHiteid == id.second) {
                result.push_back(theSimHit);
              }
            }
            LogDebug("TrkHitAssocTrace") << "by CFpos, simHit detId =  " << theSimHit.detUnitId() << " address = ("
                                         << theSimHitAddr.first << ", " << theSimHitIndex
                                         << "), process = " << theSimHit.processType() << " ("
                                         << theSimHit.eventId().bunchCrossing() << ", " << theSimHit.eventId().event()
                                         << ", " << theSimHit.trackId() << ")" << std::endl;
          }
        }
      }
    } else {  // Not a SiStripMatchedRecHit2D
      for (auto const& theSimHitAddr : simhitCFPos) {
        simHitCollectionID theSimHitCollID = theSimHitAddr.first;
        auto it = SimHitMap.find(theSimHitCollID);
        if (it != SimHitMap.end()) {
          unsigned int theSimHitIndex = theSimHitAddr.second;
          if (theSimHitIndex < (it->second).size()) {
            result.push_back((it->second)[theSimHitIndex]);
            LogDebug("TrkHitAssocTrace") << "by CFpos, simHit detId =  " << (it->second)[theSimHitIndex].detUnitId()
                                         << " address = (" << theSimHitCollID << ", " << theSimHitIndex
                                         << "), process = " << (it->second)[theSimHitIndex].processType() << " ("
                                         << (it->second)[theSimHitIndex].eventId().bunchCrossing() << ", "
                                         << (it->second)[theSimHitIndex].eventId().event() << ", "
                                         << (it->second)[theSimHitIndex].trackId() << ")" << std::endl;
          }
        }
      }
    }
    return result;
  }  // if !assocHitbySimTrack

  // Get the SimHit from the trackid instead
  auto it = SimHitMap.find(detID);
  if (it != SimHitMap.end()) {
    for (auto const& ihit : it->second) {
      unsigned int simHitid = ihit.trackId();
      EncodedEventId simHiteid = ihit.eventId();
      for (auto id : simtrackid) {
        if (simHitid == id.first && simHiteid == id.second) {
          result.push_back(ihit);
          LogDebug("TrkHitAssocTrace") << "by TrackID, simHit detId =  " << ihit.detUnitId()
                                       << ", process = " << ihit.processType() << " (" << ihit.eventId().bunchCrossing()
                                       << ", " << ihit.eventId().event() << ", " << ihit.trackId() << ")" << std::endl;
          break;
        }
      }
    }

  } else {
    /// Check if it's the gluedDet.
    auto itrphi = SimHitMap.find(detID + 2);  //iterator to the simhit in the rphi module
    auto itster = SimHitMap.find(detID + 1);  //iterator to the simhit in the stereo module
    if (itrphi != SimHitMap.end() && itster != SimHitMap.end()) {
      std::vector<PSimHit> simHitVector = itrphi->second;
      simHitVector.insert(simHitVector.end(), (itster->second).begin(), (itster->second).end());
      for (auto const& ihit : simHitVector) {
        unsigned int simHitid = ihit.trackId();
        EncodedEventId simHiteid = ihit.eventId();
        for (auto const& id : simtrackid) {
          if (simHitid == id.first && simHiteid == id.second) {
            result.push_back(ihit);
            LogDebug("TrkHitAssocTrace") << "by TrackID, simHit detId =  " << ihit.detUnitId()
                                         << ", process = " << ihit.processType() << " ("
                                         << ihit.eventId().bunchCrossing() << ", " << ihit.eventId().event() << ", "
                                         << ihit.trackId() << ")" << std::endl;
            break;
          }
        }
      }
    }
  }

  return result;
}

std::vector<SimHitIdpr> TrackerHitAssociator::associateHitId(const TrackingRecHit& thit) const {
  std::vector<SimHitIdpr> simhitid;
  associateHitId(thit, simhitid);
  return simhitid;
}

void TrackerHitAssociator::associateHitId(const TrackingRecHit& thit,
                                          std::vector<SimHitIdpr>& simtkid,
                                          std::vector<simhitAddr>* simhitCFPos) const {
  simtkid.clear();

  if (const SiTrackerMultiRecHit* rechit = dynamic_cast<const SiTrackerMultiRecHit*>(&thit))
    simtkid = associateMultiRecHitId(rechit, simhitCFPos);

  //check if it is a simple SiStripRecHit2D
  if (const SiStripRecHit2D* rechit = dynamic_cast<const SiStripRecHit2D*>(&thit))
    associateSiStripRecHit(rechit, simtkid, simhitCFPos);

  //check if it is a SiStripRecHit1D
  else if (const SiStripRecHit1D* rechit = dynamic_cast<const SiStripRecHit1D*>(&thit))
    associateSiStripRecHit(rechit, simtkid, simhitCFPos);

  //check if it is a SiStripMatchedRecHit2D
  else if (const SiStripMatchedRecHit2D* rechit = dynamic_cast<const SiStripMatchedRecHit2D*>(&thit))
    simtkid = associateMatchedRecHit(rechit, simhitCFPos);

  //check if it is a  ProjectedSiStripRecHit2D
  else if (const ProjectedSiStripRecHit2D* rechit = dynamic_cast<const ProjectedSiStripRecHit2D*>(&thit))
    simtkid = associateProjectedRecHit(rechit, simhitCFPos);

  //check if it is a Phase2TrackerRecHit1D
  else if (const Phase2TrackerRecHit1D* rechit = dynamic_cast<const Phase2TrackerRecHit1D*>(&thit))
    associatePhase2TrackerRecHit(rechit, simtkid, simhitCFPos);

  //check if it is a SiPixelRecHit
  else if (const SiPixelRecHit* rechit = dynamic_cast<const SiPixelRecHit*>(&thit))
    associatePixelRecHit(rechit, simtkid, simhitCFPos);

  //check if these are GSRecHits (from FastSim)
  if (trackerHitRTTI::isFast(thit))
    simtkid = associateFastRecHit(static_cast<const FastTrackerRecHit*>(&thit));
}

template <typename T>
inline void TrackerHitAssociator::associateSiStripRecHit(const T* simplerechit,
                                                         std::vector<SimHitIdpr>& simtrackid,
                                                         std::vector<simhitAddr>* simhitCFPos) const {
  const SiStripCluster* clust = &(*simplerechit->cluster());
  associateSimpleRecHitCluster(clust, simplerechit->geographicalId(), simtrackid, simhitCFPos);
}

//
//  Method for obtaining simTracks and simHits from a cluster
//
void TrackerHitAssociator::associateCluster(const SiStripCluster* clust,
                                            const DetId& detid,
                                            std::vector<SimHitIdpr>& simtrackid,
                                            std::vector<PSimHit>& simhit) const {
  std::vector<simhitAddr> simhitCFPos;
  associateSimpleRecHitCluster(clust, detid, simtrackid, &simhitCFPos);

  for (auto const& theSimHitAddr : simhitCFPos) {
    simHitCollectionID theSimHitCollID = theSimHitAddr.first;
    auto it = SimHitMap.find(theSimHitCollID);

    if (it != SimHitMap.end()) {
      unsigned int theSimHitIndex = theSimHitAddr.second;
      if (theSimHitIndex < (it->second).size())
        simhit.push_back((it->second)[theSimHitIndex]);
      LogDebug("TrkHitAssocTrace") << "For cluster, simHit detId =  " << (it->second)[theSimHitIndex].detUnitId()
                                   << " address = (" << theSimHitCollID << ", " << theSimHitIndex
                                   << "), process = " << (it->second)[theSimHitIndex].processType()
                                   << " (bnch, evt, trk) = (" << (it->second)[theSimHitIndex].eventId().bunchCrossing()
                                   << ", " << (it->second)[theSimHitIndex].eventId().event() << ", "
                                   << (it->second)[theSimHitIndex].trackId() << ")" << std::endl;
    }
  }
}

void TrackerHitAssociator::associateSimpleRecHitCluster(const SiStripCluster* clust,
                                                        const DetId& detid,
                                                        std::vector<SimHitIdpr>& simtrackid,
                                                        std::vector<simhitAddr>* simhitCFPos) const {
  uint32_t detID = detid.rawId();
  auto isearch = stripdigisimlink->find(detID);
  if (isearch != stripdigisimlink->end()) {  //if it is not empty
    auto link_detset = (*isearch);

    if (clust != nullptr) {  //the cluster is valid
      int clusiz = clust->amplitudes().size();
      int first = clust->firstStrip();
      int last = first + clusiz;

      LogDebug("TrkHitAssocDbg") << "Cluster size " << clusiz << " first strip = " << first
                                 << " last strip = " << last - 1 << std::endl
                                 << " detID = " << detID << " DETSET size = " << link_detset.data.size() << std::endl;
      int channel;
      for (const auto& linkiter : link_detset.data) {
        channel = (int)(linkiter.channel());
        if (channel >= first && channel < last) {
          LogDebug("TrkHitAssocDbg") << "Channel = " << std::setw(4) << linkiter.channel()
                                     << ", TrackID = " << std::setw(8) << linkiter.SimTrackId()
                                     << ", tofBin = " << std::setw(3) << linkiter.TofBin()
                                     << ", fraction = " << std::setw(8) << linkiter.fraction()
                                     << ", Position = " << linkiter.CFposition() << std::endl;
          SimHitIdpr currentId(linkiter.SimTrackId(), linkiter.eventId());

          //create a vector with the list of SimTrack ID's of the tracks that contributed to the RecHit
          //write the id only once in the vector

          if (find(simtrackid.begin(), simtrackid.end(), currentId) == simtrackid.end()) {
            LogDebug("TrkHitAssocDbg") << " Adding track id  = " << currentId.first
                                       << " Event id = " << currentId.second.event()
                                       << " Bunch Xing = " << currentId.second.bunchCrossing() << std::endl;
            simtrackid.push_back(currentId);
          }

          if (simhitCFPos != nullptr) {
            //create a vector that contains all the positions (in the MixCollection) of the SimHits that contributed to the RecHit
            //write position only once
            unsigned int currentCFPos = linkiter.CFposition();
            unsigned int tofBin = linkiter.TofBin();
            subDetTofBin theSubDetTofBin = std::make_pair(detid.subdetId(), tofBin);
            auto it = SimHitCollMap.find(theSubDetTofBin);
            if (it != SimHitCollMap.end()) {
              simhitAddr currentAddr = std::make_pair(it->second, currentCFPos);
              if (find(simhitCFPos->begin(), simhitCFPos->end(), currentAddr) == simhitCFPos->end()) {
                simhitCFPos->push_back(currentAddr);
              }
            }
          }
        }
      }
    } else {
      edm::LogError("TrackerHitAssociator") << "no cluster reference attached";
    }
  }
}

std::vector<SimHitIdpr> TrackerHitAssociator::associateMatchedRecHit(const SiStripMatchedRecHit2D* matchedrechit,
                                                                     std::vector<simhitAddr>* simhitCFPos) const {
  std::vector<SimHitIdpr> matched_mono;
  std::vector<SimHitIdpr> matched_st;

  const SiStripRecHit2D mono = matchedrechit->monoHit();
  const SiStripRecHit2D st = matchedrechit->stereoHit();
  //associate the two simple hits separately
  associateSiStripRecHit(&mono, matched_mono, simhitCFPos);
  associateSiStripRecHit(&st, matched_st, simhitCFPos);

  //save in a vector all the simtrack-id's that are common to mono and stereo hits
  std::vector<SimHitIdpr> simtrackid;
  if (!(matched_mono.empty() || matched_st.empty())) {
    for (auto const& mhit : matched_mono) {
      //save only once the ID
      if (find(simtrackid.begin(), simtrackid.end(), mhit) == simtrackid.end()) {
        //save if the stereoID matched the monoID
        if (find(matched_st.begin(), matched_st.end(), mhit) != matched_st.end()) {
          simtrackid.push_back(mhit);
        }
      }
    }
  }
  return simtrackid;
}

std::vector<SimHitIdpr> TrackerHitAssociator::associateProjectedRecHit(const ProjectedSiStripRecHit2D* projectedrechit,
                                                                       std::vector<simhitAddr>* simhitCFPos) const {
  //projectedRecHit is a "matched" rechit with only one component

  std::vector<SimHitIdpr> matched_mono;

  const SiStripRecHit2D mono = projectedrechit->originalHit();
  associateSiStripRecHit(&mono, matched_mono, simhitCFPos);
  return matched_mono;
}

void TrackerHitAssociator::associatePhase2TrackerRecHit(const Phase2TrackerRecHit1D* rechit,
                                                        std::vector<SimHitIdpr>& simtrackid,
                                                        std::vector<simhitAddr>* simhitCFPos) const {
  //
  // Phase 2 outer tracker associator
  //
  DetId detid = rechit->geographicalId();
  uint32_t detID = detid.rawId();

  auto isearch = ph2trackerdigisimlink->find(detID);
  if (isearch != ph2trackerdigisimlink->end()) {  //if it is not empty
    auto link_detset = (*isearch);
    Phase2TrackerRecHit1D::ClusterRef const& cluster = rechit->cluster();

    //check the reference is valid
    if (!(cluster.isNull())) {  //if the cluster is valid
      int minRow = (*cluster).firstStrip();
      int maxRow = (*cluster).firstStrip() + (*cluster).size();
      int Col = (*cluster).column();
      LogDebug("TrkHitAssocDbg") << "    Cluster minRow " << minRow << " maxRow " << maxRow << " column " << Col
                                 << std::endl;
      int dsl = 0;
      for (auto const& linkiter : link_detset.data) {
        ++dsl;
        std::pair<int, int> coord = Phase2TrackerDigi::channelToPixel(linkiter.channel());
        LogDebug("TrkHitAssocDbg") << "    " << dsl << ") Digi link: row " << coord.first << " col " << coord.second
                                   << std::endl;
        if (coord.first <= maxRow && coord.first >= minRow && coord.second == Col) {
          LogDebug("TrkHitAssocDbg") << "      !-> trackid   " << linkiter.SimTrackId() << endl
                                     << "          fraction  " << linkiter.fraction() << endl;
          SimHitIdpr currentId(linkiter.SimTrackId(), linkiter.eventId());
          if (find(simtrackid.begin(), simtrackid.end(), currentId) == simtrackid.end()) {
            simtrackid.push_back(currentId);
          }

          if (simhitCFPos != nullptr) {
            //create a vector that contains all the positions (in the MixCollection) of the SimHits that contributed to the RecHit
            //write position only once
            unsigned int currentCFPos = linkiter.CFposition();
            unsigned int tofBin = linkiter.TofBin();
            subDetTofBin theSubDetTofBin = std::make_pair(detid.subdetId(), tofBin);
            auto it = SimHitCollMap.find(theSubDetTofBin);
            if (it != SimHitCollMap.end()) {
              simhitAddr currentAddr = std::make_pair(it->second, currentCFPos);
              if (find(simhitCFPos->begin(), simhitCFPos->end(), currentAddr) == simhitCFPos->end()) {
                simhitCFPos->push_back(currentAddr);
              }
            }
          }
        }
      }  // end of simlink loop
    } else {
      edm::LogError("TrackerHitAssociator") << "no Phase2 outer tracker cluster reference attached";
    }
  }
}

void TrackerHitAssociator::associatePixelRecHit(const SiPixelRecHit* pixelrechit,
                                                std::vector<SimHitIdpr>& simtrackid,
                                                std::vector<simhitAddr>* simhitCFPos) const {
  //
  // Pixel associator
  //
  DetId detid = pixelrechit->geographicalId();
  uint32_t detID = detid.rawId();

  auto isearch = pixeldigisimlink->find(detID);
  if (isearch != pixeldigisimlink->end()) {  //if it is not empty
    auto link_detset = (*isearch);
    SiPixelRecHit::ClusterRef const& cluster = pixelrechit->cluster();

    //check the reference is valid

    if (!(cluster.isNull())) {  //if the cluster is valid

      int minPixelRow = (*cluster).minPixelRow();
      int maxPixelRow = (*cluster).maxPixelRow();
      int minPixelCol = (*cluster).minPixelCol();
      int maxPixelCol = (*cluster).maxPixelCol();
      LogDebug("TrkHitAssocDbg") << "    Cluster minRow " << minPixelRow << " maxRow " << maxPixelRow << std::endl
                                 << "    Cluster minCol " << minPixelCol << " maxCol " << maxPixelCol << std::endl;
      int dsl = 0;
      for (auto const& linkiter : link_detset.data) {
        ++dsl;
        std::pair<int, int> pixel_coord = PixelDigi::channelToPixel(linkiter.channel());
        LogDebug("TrkHitAssocDbg") << "    " << dsl << ") Digi link: row " << pixel_coord.first << " col "
                                   << pixel_coord.second << std::endl;
        if (pixel_coord.first <= maxPixelRow && pixel_coord.first >= minPixelRow && pixel_coord.second <= maxPixelCol &&
            pixel_coord.second >= minPixelCol) {
          LogDebug("TrkHitAssocDbg") << "      !-> trackid   " << linkiter.SimTrackId() << endl
                                     << "          fraction  " << linkiter.fraction() << endl;
          SimHitIdpr currentId(linkiter.SimTrackId(), linkiter.eventId());
          if (find(simtrackid.begin(), simtrackid.end(), currentId) == simtrackid.end()) {
            simtrackid.push_back(currentId);
          }

          if (simhitCFPos != nullptr) {
            //create a vector that contains all the positions (in the MixCollection) of the SimHits that contributed to the RecHit
            //write position only once
            unsigned int currentCFPos = linkiter.CFposition();
            unsigned int tofBin = linkiter.TofBin();
            subDetTofBin theSubDetTofBin = std::make_pair(detid.subdetId(), tofBin);
            auto it = SimHitCollMap.find(theSubDetTofBin);
            if (it != SimHitCollMap.end()) {
              simhitAddr currentAddr = std::make_pair(it->second, currentCFPos);
              if (find(simhitCFPos->begin(), simhitCFPos->end(), currentAddr) == simhitCFPos->end()) {
                simhitCFPos->push_back(currentAddr);
              }
            }
          }
        }
      }
    } else {
      edm::LogError("TrackerHitAssociator") << "no Pixel cluster reference attached";
    }
  }
}

std::vector<PSimHit> TrackerHitAssociator::associateMultiRecHit(const SiTrackerMultiRecHit* multirechit) const {
  std::vector<const TrackingRecHit*> componenthits = multirechit->recHits();
  //        std::vector<PSimHit> assimhits;
  int size = multirechit->weights().size(), idmostprobable = 0;

  for (int i = 0; i < size; ++i) {
    if (multirechit->weight(i) > multirechit->weight(idmostprobable))
      idmostprobable = i;
  }

  return associateHit(*componenthits[idmostprobable]);
}

std::vector<SimHitIdpr> TrackerHitAssociator::associateMultiRecHitId(const SiTrackerMultiRecHit* multirechit,
                                                                     std::vector<simhitAddr>* simhitCFPos) const {
  std::vector<const TrackingRecHit*> componenthits = multirechit->recHits();
  int size = multirechit->weights().size(), idmostprobable = 0;

  for (int i = 0; i < size; ++i) {
    if (multirechit->weight(i) > multirechit->weight(idmostprobable))
      idmostprobable = i;
  }

  std::vector<SimHitIdpr> simhitid;
  associateHitId(*componenthits[idmostprobable], simhitid, simhitCFPos);
  return simhitid;
}

// fastsim
std::vector<SimHitIdpr> TrackerHitAssociator::associateFastRecHit(const FastTrackerRecHit* rechit) const {
  vector<SimHitIdpr> simtrackid;
  simtrackid.clear();
  for (size_t index = 0, indexEnd = rechit->nSimTrackIds(); index < indexEnd; ++index) {
    SimHitIdpr currentId(rechit->simTrackId(index), EncodedEventId(rechit->simTrackEventId(index)));
    simtrackid.push_back(currentId);
  }
  return simtrackid;
}

inline std::string TrackerHitAssociator::printDetBnchEvtTrk(const DetId& detid,
                                                            const uint32_t& detID,
                                                            std::vector<SimHitIdpr>& simtrackid) const {
  std::stringstream message;
  message << "recHit subdet, detID = " << detid.subdetId() << ", " << detID << ", (bnch, evt, trk) = ";
  for (size_t i = 0; i < simtrackid.size(); ++i)
    message << ", (" << simtrackid[i].second.bunchCrossing() << ", " << simtrackid[i].second.event() << ", "
            << simtrackid[i].first << ")";
  // message << std::endl;
  return message.str();
}
