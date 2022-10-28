#ifndef MCTruth_CSCHitAssociator_h
#define MCTruth_CSCHitAssociator_h

#include "CondFormats/CSCObjects/interface/CSCBadChambers.h"
#include "CondFormats/DataRecord/interface/CSCBadChambersRcd.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigi.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigi.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2D.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCLayerGeometry.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

class MuonGeometryRecord;

class CSCHitAssociator {
public:
  typedef edm::DetSetVector<StripDigiSimLink> DigiSimLinks;
  typedef edm::DetSetVector<StripDigiSimLink> WireDigiSimLinks;
  typedef edm::DetSet<StripDigiSimLink> LayerLinks;
  typedef std::pair<uint32_t, EncodedEventId> SimHitIdpr;

  class Config {
  public:
    Config(const edm::ParameterSet &, edm::ConsumesCollector iC);

  private:
    friend class CSCHitAssociator;
    const edm::InputTag linksTag_;
    const edm::EDGetTokenT<DigiSimLinks> linksToken_;
    const edm::ESGetToken<CSCGeometry, MuonGeometryRecord> geomToken_;
  };

  CSCHitAssociator(const edm::Event &, const edm::EventSetup &, const Config &);

  std::vector<SimHitIdpr> associateHitId(const TrackingRecHit &) const;
  std::vector<SimHitIdpr> associateCSCHitId(const CSCRecHit2D *) const;

private:
  void initEvent(const edm::Event &, const edm::EventSetup &);

  const Config &theConfig;
  const DigiSimLinks *theDigiSimLinks;

  const CSCGeometry *cscgeom;
};

#endif
