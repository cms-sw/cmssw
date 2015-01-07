#include "SimMuon/MCTruth/interface/MuonTruth.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

MuonTruth::MuonTruth(const edm::Event& event, const edm::EventSetup& setup, const edm::ParameterSet& conf): 
  theDigiSimLinks(0),
  theWireDigiSimLinks(0),
  linksTag(conf.getParameter<edm::InputTag>("CSClinksTag")),
  wireLinksTag(conf.getParameter<edm::InputTag>("CSCwireLinksTag")),
  // CrossingFrame used or not ?
  crossingframe(conf.getParameter<bool>("crossingframe")),
  CSCsimHitsTag(conf.getParameter<edm::InputTag>("CSCsimHitsTag")),
  CSCsimHitsXFTag(conf.getParameter<edm::InputTag>("CSCsimHitsXFTag"))

{
  initEvent(event,setup);
}

MuonTruth::MuonTruth( const edm::ParameterSet& conf, edm::ConsumesCollector && iC ): 
  theDigiSimLinks(0),
  theWireDigiSimLinks(0),
  linksTag(conf.getParameter<edm::InputTag>("CSClinksTag")),
  wireLinksTag(conf.getParameter<edm::InputTag>("CSCwireLinksTag")),
  // CrossingFrame used or not ?
  crossingframe(conf.getParameter<bool>("crossingframe")),
  CSCsimHitsTag(conf.getParameter<edm::InputTag>("CSCsimHitsTag")),
  CSCsimHitsXFTag(conf.getParameter<edm::InputTag>("CSCsimHitsXFTag"))

{
  iC.consumes<DigiSimLinks>(linksTag);
  iC.consumes<DigiSimLinks>(wireLinksTag);
  if ( crossingframe ) {
    iC.consumes<CrossingFrame<PSimHit> >(CSCsimHitsXFTag);
  } else if (!CSCsimHitsTag.label().empty()){
    iC.consumes<edm::PSimHitContainer>(CSCsimHitsTag);
  }

}

void MuonTruth::initEvent(const edm::Event& event, const edm::EventSetup& setup) {

  edm::Handle<DigiSimLinks> digiSimLinks;
  LogTrace("MuonTruth") <<"getting CSC Strip DigiSimLink collection - "<<linksTag;
  event.getByLabel(linksTag, digiSimLinks);
  theDigiSimLinks = digiSimLinks.product();

  edm::Handle<DigiSimLinks> wireDigiSimLinks;
  LogTrace("MuonTruth") <<"getting CSC Wire DigiSimLink collection - "<<wireLinksTag;
  event.getByLabel(wireLinksTag, wireDigiSimLinks);
  theWireDigiSimLinks = wireDigiSimLinks.product();

  // get CSC Geometry to use CSCLayer methods
  edm::ESHandle<CSCGeometry> mugeom;
  setup.get<MuonGeometryRecord>().get( mugeom );
  cscgeom = &*mugeom;

  // get CSC Bad Chambers (ME4/2)
  edm::ESHandle<CSCBadChambers> badChambers;
  setup.get<CSCBadChambersRcd>().get(badChambers);
  cscBadChambers = badChambers.product();

  theSimHitMap.clear();

  if (crossingframe) {
    
    edm::Handle<CrossingFrame<PSimHit> > cf;
    LogTrace("MuonTruth") <<"getting CrossingFrame<PSimHit> collection - "<<CSCsimHitsXFTag;
    event.getByLabel(CSCsimHitsXFTag, cf);
    
    std::auto_ptr<MixCollection<PSimHit> > 
      CSCsimhits( new MixCollection<PSimHit>(cf.product()) );
    LogTrace("MuonTruth") <<"... size = "<<CSCsimhits->size();

    for(MixCollection<PSimHit>::MixItr hitItr = CSCsimhits->begin();
	hitItr != CSCsimhits->end(); ++hitItr) 
      {
	theSimHitMap[hitItr->detUnitId()].push_back(*hitItr);
      }
    
  } else if (!CSCsimHitsTag.label().empty()){

    edm::Handle<edm::PSimHitContainer> CSCsimhits;
    LogTrace("MuonTruth") <<"getting PSimHit collection - "<<CSCsimHitsTag;
    event.getByLabel(CSCsimHitsTag, CSCsimhits);    
    LogTrace("MuonTruth") <<"... size = "<<CSCsimhits->size();
    
    for(edm::PSimHitContainer::const_iterator hitItr = CSCsimhits->begin();
	hitItr != CSCsimhits->end(); ++hitItr)
      {
	theSimHitMap[hitItr->detUnitId()].push_back(*hitItr);
      }
  }
}

float MuonTruth::muonFraction()
{
  if(theChargeMap.size() == 0) return 0.;

  float muonCharge = 0.;
  for(std::map<SimHitIdpr, float>::const_iterator chargeMapItr = theChargeMap.begin();
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
  for(std::map<SimHitIdpr, float>::const_iterator chargeMapItr = theChargeMap.begin();
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



std::vector<PSimHit> MuonTruth::hitsFromSimTrack(MuonTruth::SimHitIdpr truthId)
{
  std::vector<PSimHit> result;

  auto found = theSimHitMap.find(theDetId);
  if (found != theSimHitMap.end()) {

    for(auto const& hit: found->second)
      {
        unsigned int hitTrack = hit.trackId();
        EncodedEventId hitEvId = hit.eventId();
        
        if(hitTrack == truthId.first && hitEvId == truthId.second) 
          {
            result.push_back(hit);
          }
      }
  }
  return result;
}


int MuonTruth::particleType(MuonTruth::SimHitIdpr truthId)
{
  int result = 0;
  const std::vector<PSimHit>& hits = hitsFromSimTrack(truthId);
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

  int nchannels = recHit.nStrips();
  const CSCLayerGeometry * laygeom = cscgeom->layer(recHit.cscDetId())->geometry();

  for(int idigi = 0; idigi < nchannels; ++idigi)
  {
    // strip and readout channel numbers may differ in ME1/1A
    int istrip = recHit.channels(idigi);
    int channel = laygeom->channel(istrip);
    float weight = recHit.adcs(idigi,0);//DL: I think this is wrong before and after...seems to assume one time binadcContainer[idigi];

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
      SimHitIdpr truthId(linkItr->SimTrackId(),linkItr->eventId());
      std::map<SimHitIdpr, float>::const_iterator chargeMapItr = theChargeMap.find(truthId);
      if(chargeMapItr == theChargeMap.end())
      {
	theChargeMap[truthId] = charge;
      }
      else
      {
	theChargeMap[truthId] += charge;
      }
    }
  }
}
    

