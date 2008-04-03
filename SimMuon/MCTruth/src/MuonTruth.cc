#include "SimMuon/MCTruth/interface/MuonTruth.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

MuonTruth::MuonTruth()
: theSimTrackContainer(0),
  theDigiSimLinks(0),
  theWireDigiSimLinks(0),
  theSimHitMap("MuonCSCHits")
{
}

void MuonTruth::eventSetup(const edm::Event & event)
{
  edm::Handle<edm::SimTrackContainer> simTrackCollection;
  event.getByLabel("g4SimHits", simTrackCollection);
  theSimTrackContainer = simTrackCollection.product();

  edm::Handle<DigiSimLinks> digiSimLinks;
  edm::InputTag linksTag("muonCSCDigis" , "MuonCSCStripDigiSimLinks");
  event.getByLabel(linksTag, digiSimLinks);
  theDigiSimLinks = digiSimLinks.product();

  edm::Handle<DigiSimLinks> wireDigiSimLinks;
  edm::InputTag wireLinksTag("muonCSCDigis" , "MuonCSCWireDigiSimLinks");
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


std::vector<const PSimHit *> MuonTruth::simHits()
{
  std::vector<const PSimHit *> result;
  for(std::map<int, float>::const_iterator chargeMapItr = theChargeMap.begin();
      chargeMapItr != theChargeMap.end(); ++chargeMapItr)
  {
    std::vector<const PSimHit *> trackHits = hitsFromSimTrack(chargeMapItr->first);
    result.insert(result.end(), trackHits.begin(), trackHits.end());
  }
  return result;
}


std::vector<const PSimHit *> MuonTruth::muonHits()
{
  std::vector<const PSimHit *> result;
  std::vector<const PSimHit *> allHits = simHits();
  std::vector<const PSimHit *>::const_iterator hitItr = allHits.begin(), lastHit = allHits.end();

  for( ; hitItr != lastHit; ++hitItr)
  {
    if(abs((**hitItr).particleType()) == 13)
    {
      result.push_back(*hitItr);
    }
  }
  return result;
}


std::vector<MuonTruth::SimHitIdpr> MuonTruth::associateHitId(const TrackingRecHit & hit)
{
  //  LogTrace("MuonAssociatorByHits")<<"CSCassociateHitId";
  std::vector<SimHitIdpr> simtrackids;
  simtrackids.clear();
  const TrackingRecHit * hitp = &hit;
  const CSCRecHit2D * cscrechit = dynamic_cast<const CSCRecHit2D *>(hitp);

  if (cscrechit) {
    //    LogTrace("MuonAssociatorByHits")<<"cscrechit : "<<*cscrechit;
    analyze(*cscrechit);
    std::vector<const PSimHit *> matchedSimHits = simHits();
    //    LogTrace("MuonAssociatorByHits")<<"matchedSimHits.size() = "<<matchedSimHits.size();
    for(std::vector<const PSimHit *>::const_iterator hIT=matchedSimHits.begin(); hIT != matchedSimHits.end(); hIT++) {
      SimHitIdpr currentId((*hIT)->trackId(), (*hIT)->eventId());
      simtrackids.push_back(currentId);
    }
  } else {
    edm::LogWarning("MuonAssociatorByHits")<<"WARNING in CSCassociateHitId, null dynamic_cast !";
  }
  return simtrackids;
}



std::vector<const PSimHit *> MuonTruth::hitsFromSimTrack(int index) const
{
  std::vector<const PSimHit *> result;
  edm::PSimHitContainer hits = theSimHitMap.hits(theDetId);
  edm::PSimHitContainer::const_iterator hitItr = hits.begin(), lastHit = hits.end();

  for( ; hitItr != lastHit; ++hitItr)
  {
    int hitTrack = hitItr->trackId();
    if(hitTrack == index) 
    {
      result.push_back(&*hitItr);
    }
  }
  return result;
}


int MuonTruth::particleType(int simTrack) const
{
  int result = 0;
  std::vector<const PSimHit *> hits = hitsFromSimTrack(simTrack);
  if(!hits.empty())
  {
    result = hits[0]->particleType();
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
//
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
    

