#ifndef MCTruth_MuonTruth_h
#define MCTruth_MuonTruth_h

#include "DataFormats/CSCRecHit/interface/CSCRecHit2D.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigi.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "SimMuon/MCTruth/interface/PSimHitMap.h"


class MuonTruth
{
public:
  typedef edm::DetSetVector<StripDigiSimLink> DigiSimLinks;
  typedef edm::DetSet<StripDigiSimLink> LayerLinks;

  MuonTruth();

  void eventSetup(const edm::Event & event);

  void analyze(const CSCRecHit2D & recHit);
  void analyze(const CSCStripDigi & stripDigi, int layerId);

  /// analyze() must be called before any of the following
  float muonFraction();

  std::vector<const PSimHit *> muonHits();

  std::vector<const PSimHit *> simHits();

private:

  std::vector<const PSimHit *> hitsFromSimTrack(int simTrack) const;

  // goes to SimHits for information
  int particleType(int simTrack) const;

  void addChannel(const LayerLinks &layerLinks, int channel, float weight=1.);

  std::map<int, float> theChargeMap;
  float theTotalCharge;
  int theDetId;

  const edm::SimTrackContainer * theSimTrackContainer;

  const DigiSimLinks  * theDigiSimLinks;

  PSimHitMap theSimHitMap;
};

#endif

