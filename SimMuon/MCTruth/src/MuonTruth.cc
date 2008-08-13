#include "SimMuon/MCTruth/interface/MuonTruth.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

MuonTruth::MuonTruth(const edm::ParameterSet& conf)
: theDigiSimLinks(0),
  theWireDigiSimLinks(0),
  theSimHitMap(conf.getParameter<edm::InputTag>("CSCsimHitsXFTag")),
  simTracksXFTag(conf.getParameter<edm::InputTag>("simtracksXFTag")),
  linksTag(conf.getParameter<edm::InputTag>("CSClinksTag")),
  wireLinksTag(conf.getParameter<edm::InputTag>("CSCwireLinksTag"))
{
}

void MuonTruth::eventSetup(const edm::Event & event)
{
  edm::Handle<CrossingFrame<SimTrack> > xFrame;
  LogTrace("MuonTruth") <<"getting CrossingFrame<SimTrack> collection - "<<simTracksXFTag;
  event.getByLabel(simTracksXFTag,xFrame);
  std::auto_ptr<MixCollection<SimTrack> > 
    theSimTracks( new MixCollection<SimTrack>(xFrame.product()) );

  edm::Handle<DigiSimLinks> digiSimLinks;
  LogTrace("MuonTruth") <<"getting CSC Strip DigiSimLink collection - "<<linksTag;
  event.getByLabel(linksTag, digiSimLinks);
  theDigiSimLinks = digiSimLinks.product();

  edm::Handle<DigiSimLinks> wireDigiSimLinks;
  LogTrace("MuonTruth") <<"getting CSC Wire DigiSimLink collection - "<<wireLinksTag;
  event.getByLabel(wireLinksTag, wireDigiSimLinks);
  theWireDigiSimLinks = wireDigiSimLinks.product();

  theSimHitMap.fill(event);
}

float MuonTruth::muonFraction()
{
  if(theChargeMap.size() == 0) return 0.;

  float muonCharge = 0.;
  for(std::map<int, float>::const_iterator chargeMapItr = theChargeMap.begin();
      chargeMapItr != theChargeMap.end(); ++chargeMapItr)
  {
    if( abs(particleType(chargeMapItr->first)) == 13)
    {
      muonCharge += chargeMapItr->second;
    }
  }

  return muonCharge / theTotalCharge;
}


std::vector<PSimHit> MuonTruth::simHits()
{
  std::vector<PSimHit> result;
  for(std::map<int, float>::const_iterator chargeMapItr = theChargeMap.begin();
      chargeMapItr != theChargeMap.end(); ++chargeMapItr)
  {
    std::vector<PSimHit> trackHits = hitsFromSimTrack(chargeMapItr->first);
    result.insert(result.end(), trackHits.begin(), trackHits.end());
  }
  
  return result;
}


std::vector<PSimHit> MuonTruth::muonHits()
{
  std::vector<PSimHit> result;
  std::vector<PSimHit> allHits = simHits();
  std::vector<PSimHit>::const_iterator hitItr = allHits.begin(), lastHit = allHits.end();

  for( ; hitItr != lastHit; ++hitItr)
  {
    if(abs((*hitItr).particleType()) == 13)
    {
      result.push_back(*hitItr);
    }
  }
  return result;
}


std::vector<MuonTruth::SimHitIdpr> MuonTruth::associateHitId(const TrackingRecHit & hit)
{
  std::vector<SimHitIdpr> simtrackids;
  simtrackids.clear();
  const TrackingRecHit * hitp = &hit;
  const CSCRecHit2D * cscrechit = dynamic_cast<const CSCRecHit2D *>(hitp);

  if (cscrechit) {
    analyze(*cscrechit);
    std::vector<PSimHit> matchedSimHits = simHits();
    for(std::vector<PSimHit>::const_iterator hIT=matchedSimHits.begin(); hIT != matchedSimHits.end(); hIT++) {
      SimHitIdpr currentId(hIT->trackId(), hIT->eventId());
      simtrackids.push_back(currentId);
    }
  } else {
    edm::LogWarning("MuonTruth")<<"WARNING in MuonTruth::associateHitId, null dynamic_cast !";
  }
  return simtrackids;
}


std::vector<PSimHit> MuonTruth::hitsFromSimTrack(int index)
{
  std::vector<PSimHit> result;
  edm::PSimHitContainer hits = theSimHitMap.hits(theDetId);
  edm::PSimHitContainer::const_iterator hitItr = hits.begin(), lastHit = hits.end();

  for( ; hitItr != lastHit; ++hitItr)
  {
    int hitTrack = hitItr->trackId();
    if(hitTrack == index) 
    {
      result.push_back(*hitItr);
    }
  }
  return result;
}


int MuonTruth::particleType(int simTrack)
{
  int result = 0;
  std::vector<PSimHit> hits = hitsFromSimTrack(simTrack);
  if(!hits.empty())
  {
    result = hits[0].particleType();
  }
  return result;
}


void MuonTruth::analyze(const CSCRecHit2D & recHit)
{
  theChargeMap.clear();
  theTotalCharge = 0.;
  theDetId = recHit.geographicalId().rawId();

  int nchannels = recHit.channels().size();
  CSCRecHit2D::ADCContainer adcContainer = recHit.adcs();
  for(int idigi = 0; idigi < nchannels; ++idigi)
  {
    int channel = recHit.channels()[idigi];
    float weight = adcContainer[idigi];

    DigiSimLinks::const_iterator layerLinks = theDigiSimLinks->find(theDetId);

    if(layerLinks != theDigiSimLinks->end())
    {
      addChannel(*layerLinks, channel, weight);
    }
  }
}


void MuonTruth::analyze(const CSCStripDigi & stripDigi, int rawDetIdCorrespondingToCSCLayer)
{
  theDetId = rawDetIdCorrespondingToCSCLayer;
  theChargeMap.clear();
  theTotalCharge = 0.;

  DigiSimLinks::const_iterator layerLinks = theDigiSimLinks->find(theDetId);
  if(layerLinks != theDigiSimLinks->end())
  {
    addChannel(*layerLinks, stripDigi.getStrip(), 1.);
  }
}


void MuonTruth::analyze(const CSCWireDigi & wireDigi, int rawDetIdCorrespondingToCSCLayer) 
{
  theDetId = rawDetIdCorrespondingToCSCLayer;
  theChargeMap.clear();
  theTotalCharge = 0.;

  WireDigiSimLinks::const_iterator layerLinks = theWireDigiSimLinks->find(theDetId);

  if(layerLinks != theDigiSimLinks->end()) 
  {
    // In the simulation digis, the channel labels for wires and strips must be distinct, therefore:
    int wireDigiInSimulation = wireDigi.getWireGroup() + 100;
    //
    addChannel(*layerLinks, wireDigiInSimulation, 1.);
  }
}


void MuonTruth::addChannel(const LayerLinks &layerLinks, int channel, float weight)
{
  LayerLinks::const_iterator linkItr = layerLinks.begin(), 
                             lastLayerLink = layerLinks.end();

  for ( ; linkItr != lastLayerLink; ++linkItr)
  {
    int linkChannel = linkItr->channel();
    if(linkChannel == channel)
    {
      float charge = linkItr->fraction() * weight;
      theTotalCharge += charge;
      // see if it's in the map
      int simTrack = linkItr->SimTrackId();
      std::map<int, float>::const_iterator chargeMapItr = theChargeMap.find( simTrack );
      if(chargeMapItr == theChargeMap.end())
      {
	theChargeMap[simTrack] = charge;
      }
      else
      {
	theChargeMap[simTrack] += charge;
      }
    }
  }
}
    

