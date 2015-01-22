#ifndef MCTruth_MuonTruth_h
#define MCTruth_MuonTruth_h

#include "DataFormats/CSCRecHit/interface/CSCRecHit2D.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigi.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigi.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCLayerGeometry.h"
#include "CondFormats/CSCObjects/interface/CSCBadChambers.h"
#include "CondFormats/DataRecord/interface/CSCBadChambersRcd.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

class MuonTruth
{
public:
  typedef edm::DetSetVector<StripDigiSimLink> DigiSimLinks;
  typedef edm::DetSetVector<StripDigiSimLink> WireDigiSimLinks;
  typedef edm::DetSet<StripDigiSimLink> LayerLinks;
  typedef std::pair <uint32_t, EncodedEventId> SimHitIdpr;

  MuonTruth(const edm::Event&, const edm::EventSetup&, const edm::ParameterSet&); 
  MuonTruth(const edm::ParameterSet&, edm::ConsumesCollector && iC);
 
  void initEvent(const edm::Event &, const edm::EventSetup& );

  void analyze(const CSCRecHit2D & recHit);
  void analyze(const CSCStripDigi & stripDigi, int rawDetIdCorrespondingToCSCLayer);
  void analyze(const CSCWireDigi & wireDigi  , int rawDetIdCorrespondingToCSCLayer);

  /// analyze() must be called before any of the following
  float muonFraction();

  std::vector<PSimHit> muonHits();

  std::vector<PSimHit> simHits();

  const CSCBadChambers* cscBadChambers;

private:

  std::vector<PSimHit> hitsFromSimTrack(SimHitIdpr truthId) ;
  // goes to SimHits for information
  int particleType(SimHitIdpr truthId);

  void addChannel(const LayerLinks &layerLinks, int channel, float weight=1.);

  std::map<SimHitIdpr, float> theChargeMap;
  float theTotalCharge;

  unsigned int theDetId;

  const DigiSimLinks  * theDigiSimLinks;
  const DigiSimLinks  * theWireDigiSimLinks;

  edm::InputTag linksTag;
  edm::InputTag wireLinksTag;

  bool crossingframe;
  edm::InputTag CSCsimHitsTag;
  edm::InputTag CSCsimHitsXFTag;

  std::map<unsigned int, edm::PSimHitContainer> theSimHitMap;

  const CSCGeometry* cscgeom;
};

#endif
