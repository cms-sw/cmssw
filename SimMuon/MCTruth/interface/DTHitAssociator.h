#ifndef MCTruth_DTHitAssociator_h
#define MCTruth_DTHitAssociator_h

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecHit1D.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "SimDataFormats/DigiSimLinks/interface/DTDigiSimLinkCollection.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"

#include <map>
#include <vector>

class MuonGeometryRecord;

class DTHitAssociator {
public:
  typedef std::pair<uint32_t, EncodedEventId> SimHitIdpr;
  typedef std::pair<PSimHit, bool> PSimHit_withFlag;
  typedef std::map<DTWireId, std::vector<PSimHit_withFlag>> SimHitMap;
  typedef std::map<DTWireId, std::vector<DTRecHit1DPair>> RecHitMap;
  typedef std::map<DTWireId, std::vector<DTDigi>> DigiMap;
  typedef std::map<DTWireId, std::vector<DTDigiSimLink>> LinksMap;

  class Config {
  public:
    Config(const edm::ParameterSet &, edm::ConsumesCollector iC);

  private:
    friend class DTHitAssociator;

    edm::InputTag DTsimhitsTag;
    edm::InputTag DTsimhitsXFTag;
    edm::InputTag DTdigiTag;
    edm::InputTag DTdigisimlinkTag;
    edm::InputTag DTrechitTag;

    edm::EDGetTokenT<edm::PSimHitContainer> DTsimhitsToken;
    edm::EDGetTokenT<CrossingFrame<PSimHit>> DTsimhitsXFToken;
    edm::EDGetTokenT<DTDigiCollection> DTdigiToken;
    edm::EDGetTokenT<DTDigiSimLinkCollection> DTdigisimlinkToken;
    edm::EDGetTokenT<DTRecHitCollection> DTrechitToken;

    edm::ESGetToken<DTGeometry, MuonGeometryRecord> geomToken;

    bool dumpDT;
    bool crossingframe;
    bool links_exist;
    bool associatorByWire;
  };

  DTHitAssociator(const edm::Event &, const edm::EventSetup &, const Config &, bool printRtS);

  std::vector<SimHitIdpr> associateHitId(const TrackingRecHit &hit) const;
  std::vector<SimHitIdpr> associateDTHitId(const DTRecHit1D *dtrechit) const;

  std::vector<PSimHit> associateHit(const TrackingRecHit &hit) const;

  SimHitMap mapOfSimHit;
  RecHitMap mapOfRecHit;
  DigiMap mapOfDigi;
  LinksMap mapOfLinks;

private:
  void initEvent(const edm::Event &, const edm::EventSetup &);

  bool SimHitOK(const edm::ESHandle<DTGeometry> &, const PSimHit &);
  Config const &config_;
  bool printRtS;
};

#endif
