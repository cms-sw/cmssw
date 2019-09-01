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

class CSCHitAssociator {
public:
  typedef edm::DetSetVector<StripDigiSimLink> DigiSimLinks;
  typedef edm::DetSetVector<StripDigiSimLink> WireDigiSimLinks;
  typedef edm::DetSet<StripDigiSimLink> LayerLinks;
  typedef std::pair<uint32_t, EncodedEventId> SimHitIdpr;

  CSCHitAssociator(const edm::Event &, const edm::EventSetup &, const edm::ParameterSet &);
  CSCHitAssociator(const edm::ParameterSet &, edm::ConsumesCollector &&iC);

  void initEvent(const edm::Event &, const edm::EventSetup &);

  std::vector<SimHitIdpr> associateHitId(const TrackingRecHit &) const;
  std::vector<SimHitIdpr> associateCSCHitId(const CSCRecHit2D *) const;

private:
  const DigiSimLinks *theDigiSimLinks;

  edm::InputTag linksTag;

  const CSCGeometry *cscgeom;
};

#endif
