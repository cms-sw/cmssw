#include "SimMuon/MCTruth/interface/GEMHitAssociator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

// Constructor
GEMHitAssociator::Config::Config(const edm::ParameterSet &conf, edm::ConsumesCollector iC)
    : GEMdigisimlinkTag(conf.getParameter<edm::InputTag>("GEMdigisimlinkTag")),
      // CrossingFrame used or not ?
      GEMsimhitsTag(conf.getParameter<edm::InputTag>("GEMsimhitsTag")),
      GEMsimhitsXFTag(conf.getParameter<edm::InputTag>("GEMsimhitsXFTag")),
      crossingframe(conf.getParameter<bool>("crossingframe")),
      useGEMs_(conf.getParameter<bool>("useGEMs")) {
  if (crossingframe) {
    GEMsimhitsXFToken_ = iC.consumes<CrossingFrame<PSimHit>>(GEMsimhitsXFTag);
  } else if (!GEMsimhitsTag.label().empty()) {
    GEMsimhitsToken_ = iC.consumes<edm::PSimHitContainer>(GEMsimhitsTag);
  }

  GEMdigisimlinkToken_ = iC.consumes<edm::DetSetVector<GEMDigiSimLink>>(GEMdigisimlinkTag);
}

GEMHitAssociator::GEMHitAssociator(const edm::Event &e, const Config &config) : theConfig(config) { initEvent(e); }

void GEMHitAssociator::initEvent(const edm::Event &e) {
  if (theConfig.useGEMs_) {
    if (theConfig.crossingframe) {
      LogTrace("GEMHitAssociator") << "getting CrossingFrame<PSimHit> collection - " << theConfig.GEMsimhitsXFTag;
      CrossingFrame<PSimHit> const &cf = e.get(theConfig.GEMsimhitsXFToken_);

      std::unique_ptr<MixCollection<PSimHit>> GEMsimhits(new MixCollection<PSimHit>(&cf));
      LogTrace("GEMHitAssociator") << "... size = " << GEMsimhits->size();

      //   MixCollection<PSimHit> & simHits = *hits;

      for (MixCollection<PSimHit>::MixItr hitItr = GEMsimhits->begin(); hitItr != GEMsimhits->end(); ++hitItr) {
        _SimHitMap[hitItr->detUnitId()].push_back(*hitItr);
      }

    } else if (!theConfig.GEMsimhitsTag.label().empty()) {
      LogTrace("GEMHitAssociator") << "getting PSimHit collection - " << theConfig.GEMsimhitsTag;
      edm::PSimHitContainer const &GEMsimhits = e.get(theConfig.GEMsimhitsToken_);
      LogTrace("GEMHitAssociator") << "... size = " << GEMsimhits.size();

      // arrange the hits by detUnit
      for (edm::PSimHitContainer::const_iterator hitItr = GEMsimhits.begin(); hitItr != GEMsimhits.end(); ++hitItr) {
        _SimHitMap[hitItr->detUnitId()].push_back(*hitItr);
      }
    }

    LogTrace("GEMHitAssociator") << "getting GEM Strip DigiSimLink collection - " << theConfig.GEMdigisimlinkTag;
    theDigiSimLinks = &e.get(theConfig.GEMdigisimlinkToken_);
  }
}
// end of constructor

std::vector<GEMHitAssociator::SimHitIdpr> GEMHitAssociator::associateRecHit(const GEMRecHit *gemrechit) const {
  std::vector<SimHitIdpr> matched;

  if (theConfig.useGEMs_) {
    if (gemrechit) {
      GEMDetId gemDetId = gemrechit->gemId();
      int fstrip = gemrechit->firstClusterStrip();
      int cls = gemrechit->clusterSize();
      // int bx = gemrechit->BunchX();

      DigiSimLinks::const_iterator layerLinks = theDigiSimLinks->find(gemDetId);

      if (layerLinks != theDigiSimLinks->end()) {
        for (int i = fstrip; i < (fstrip + cls); ++i) {
          for (LayerLinks::const_iterator itlink = layerLinks->begin(); itlink != layerLinks->end(); ++itlink) {
            int ch = static_cast<int>(itlink->getStrip());
            if (ch != i)
              continue;

            SimHitIdpr currentId(itlink->getTrackId(), itlink->getEventId());
            if (find(matched.begin(), matched.end(), currentId) == matched.end())
              matched.push_back(currentId);
          }
        }

      } else
        edm::LogWarning("GEMHitAssociator")
            << "*** WARNING in GEMHitAssociator: GEM layer " << gemDetId << " has no DigiSimLinks !" << std::endl;

    } else
      edm::LogWarning("GEMHitAssociator") << "*** WARNING in GEMHitAssociator::associateRecHit, null "
                                             "dynamic_cast !";
  }

  return matched;
}
