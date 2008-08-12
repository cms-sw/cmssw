#ifndef MCTruth_MuonTruth_h
#define MCTruth_MuonTruth_h

#include "DataFormats/CSCRecHit/interface/CSCRecHit2D.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigi.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigi.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "SimMuon/MCTruth/interface/PSimHitMap.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"


class MuonTruth
{
public:
  typedef edm::DetSetVector<StripDigiSimLink> DigiSimLinks;
  typedef edm::DetSetVector<StripDigiSimLink> WireDigiSimLinks;
  typedef edm::DetSet<StripDigiSimLink> LayerLinks;
  typedef std::pair <uint32_t, EncodedEventId> SimHitIdpr;

  MuonTruth(const edm::ParameterSet&);

  void eventSetup(const edm::Event & event);

  void analyze(const CSCRecHit2D & recHit);
  void analyze(const CSCStripDigi & stripDigi, int rawDetIdCorrespondingToCSCLayer);
  void analyze(const CSCWireDigi & wireDigi  , int rawDetIdCorrespondingToCSCLayer);

  /// analyze() must be called before any of the following
  float muonFraction();

  std::vector<PSimHit> muonHits();

  std::vector<PSimHit> simHits();

  std::vector<SimHitIdpr> associateHitId(const TrackingRecHit &);

private:

  std::vector<PSimHit> hitsFromSimTrack(int simTrack);
  // goes to SimHits for information
  int particleType(int simTrack);

  void addChannel(const LayerLinks &layerLinks, int channel, float weight=1.);

  std::map<int, float> theChargeMap;
  float theTotalCharge;
  int theDetId;

  const MixCollection<SimTrack> * theSimTracks;

  const DigiSimLinks  * theDigiSimLinks;
  const DigiSimLinks  * theWireDigiSimLinks;

  PSimHitMap theSimHitMap;

  edm::InputTag simTracksXFTag;
  edm::InputTag linksTag;
  edm::InputTag wireLinksTag;
};

#endif

