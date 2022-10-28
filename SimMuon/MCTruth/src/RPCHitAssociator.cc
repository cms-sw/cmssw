#include "SimMuon/MCTruth/interface/RPCHitAssociator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

// Constructor
RPCHitAssociator::Config::Config(const edm::ParameterSet &conf, edm::ConsumesCollector iC)
    : RPCdigisimlinkTag(conf.getParameter<edm::InputTag>("RPCdigisimlinkTag")),
      // CrossingFrame used or not ?
      RPCsimhitsTag(conf.getParameter<edm::InputTag>("RPCsimhitsTag")),
      RPCsimhitsXFTag(conf.getParameter<edm::InputTag>("RPCsimhitsXFTag")),
      crossingframe(conf.getParameter<bool>("crossingframe")) {
  if (crossingframe) {
    RPCsimhitsXFToken_ = iC.consumes<CrossingFrame<PSimHit>>(RPCsimhitsXFTag);
  } else if (!RPCsimhitsTag.label().empty()) {
    RPCsimhitsToken_ = iC.consumes<edm::PSimHitContainer>(RPCsimhitsTag);
  }

  RPCdigisimlinkToken_ = iC.consumes<edm::DetSetVector<RPCDigiSimLink>>(RPCdigisimlinkTag);
}

RPCHitAssociator::RPCHitAssociator(const edm::Event &e, const Config &conf) : theConfig(conf) { initEvent(e); }

void RPCHitAssociator::initEvent(const edm::Event &e)

{
  if (theConfig.crossingframe) {
    LogTrace("RPCHitAssociator") << "getting CrossingFrame<PSimHit> collection - " << theConfig.RPCsimhitsXFTag;
    CrossingFrame<PSimHit> const &cf = e.get(theConfig.RPCsimhitsXFToken_);

    std::unique_ptr<MixCollection<PSimHit>> RPCsimhits(new MixCollection<PSimHit>(&cf));
    LogTrace("RPCHitAssociator") << "... size = " << RPCsimhits->size();

    //   MixCollection<PSimHit> & simHits = *hits;

    for (MixCollection<PSimHit>::MixItr hitItr = RPCsimhits->begin(); hitItr != RPCsimhits->end(); ++hitItr) {
      _SimHitMap[hitItr->detUnitId()].push_back(*hitItr);
    }

  } else if (!theConfig.RPCsimhitsTag.label().empty()) {
    LogTrace("RPCHitAssociator") << "getting PSimHit collection - " << theConfig.RPCsimhitsTag;
    edm::PSimHitContainer const &RPCsimhits = e.get(theConfig.RPCsimhitsToken_);
    LogTrace("RPCHitAssociator") << "... size = " << RPCsimhits.size();

    // arrange the hits by detUnit
    for (edm::PSimHitContainer::const_iterator hitItr = RPCsimhits.begin(); hitItr != RPCsimhits.end(); ++hitItr) {
      _SimHitMap[hitItr->detUnitId()].push_back(*hitItr);
    }
  }

  LogTrace("RPCHitAssociator") << "getting RPCDigiSimLink collection - " << theConfig.RPCdigisimlinkTag;
  _thelinkDigis = e.getHandle(theConfig.RPCdigisimlinkToken_);
}
// end of constructor

std::vector<RPCHitAssociator::SimHitIdpr> RPCHitAssociator::associateRecHit(const TrackingRecHit &hit) const {
  std::vector<SimHitIdpr> matched;

  const TrackingRecHit *hitp = &hit;
  const RPCRecHit *rpcrechit = dynamic_cast<const RPCRecHit *>(hitp);

  if (rpcrechit) {
    RPCDetId rpcDetId = rpcrechit->rpcId();
    int fstrip = rpcrechit->firstClusterStrip();
    int cls = rpcrechit->clusterSize();
    int bx = rpcrechit->BunchX();

    for (int i = fstrip; i < fstrip + cls; ++i) {
      std::set<RPCDigiSimLink> links = findRPCDigiSimLink(rpcDetId.rawId(), i, bx);

      if (links.empty())
        LogTrace("RPCHitAssociator") << "*** WARNING in RPCHitAssociator::associateRecHit, RPCRecHit " << *rpcrechit
                                     << ", strip " << i << " has no associated RPCDigiSimLink !" << endl;

      for (std::set<RPCDigiSimLink>::iterator itlink = links.begin(); itlink != links.end(); ++itlink) {
        SimHitIdpr currentId(itlink->getTrackId(), itlink->getEventId());
        if (find(matched.begin(), matched.end(), currentId) == matched.end())
          matched.push_back(currentId);
      }
    }

  } else
    LogTrace("RPCHitAssociator") << "*** WARNING in RPCHitAssociator::associateRecHit, null "
                                    "dynamic_cast !";

  return matched;
}

std::set<RPCDigiSimLink> RPCHitAssociator::findRPCDigiSimLink(uint32_t rpcDetId, int strip, int bx) const {
  std::set<RPCDigiSimLink> links;

  for (edm::DetSetVector<RPCDigiSimLink>::const_iterator itlink = _thelinkDigis->begin();
       itlink != _thelinkDigis->end();
       itlink++) {
    for (edm::DetSet<RPCDigiSimLink>::const_iterator digi_iter = itlink->data.begin(); digi_iter != itlink->data.end();
         ++digi_iter) {
      uint32_t detid = digi_iter->getDetUnitId();
      int str = digi_iter->getStrip();
      int bunchx = digi_iter->getBx();

      if (detid == rpcDetId && str == strip && bunchx == bx) {
        links.insert(*digi_iter);
      }
    }
  }

  return links;
}
