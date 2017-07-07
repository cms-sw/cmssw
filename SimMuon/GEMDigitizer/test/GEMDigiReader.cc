/** \class GEMDigiReader
 *
 *  Dumps GEM digis 
 *  
 *  \authors: Vadim Khotilovich
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include <map>
#include <set>

#include "DataFormats/Common/interface/DetSet.h"

#include <iostream>

using namespace std;


class GEMDigiReader: public edm::EDAnalyzer
{
public:

  explicit GEMDigiReader(const edm::ParameterSet& pset);
  
  virtual ~GEMDigiReader(){}
  
  void analyze(const edm::Event &, const edm::EventSetup&); 
  
private:

  edm::EDGetTokenT<edm::PSimHitContainer> simhitToken_;
  edm::EDGetTokenT<GEMDigiCollection> gemDigiToken_;
  edm::EDGetTokenT<edm::DetSetVector<StripDigiSimLink> > gemDigiSimLinkToken_;
};



GEMDigiReader::GEMDigiReader(const edm::ParameterSet& pset) :
  simhitToken_(consumes<edm::PSimHitContainer>(pset.getParameter<edm::InputTag>("simhitToken"))),
  gemDigiToken_(consumes<GEMDigiCollection>(pset.getParameter<edm::InputTag>("gemDigiToken"))),
  gemDigiSimLinkToken_(consumes<edm::DetSetVector<StripDigiSimLink> >(pset.getParameter<edm::InputTag>("gemDigiSimLinkToken")))
{
}


void GEMDigiReader::analyze(const edm::Event & event, const edm::EventSetup& eventSetup)
{
  LogDebug("GEMDigiReader") << "--- Run: " << event.id().run() << " Event: " << event.id().event() << endl;

  edm::ESHandle<GEMGeometry> pDD;
  eventSetup.get<MuonGeometryRecord>().get( pDD );

  edm::Handle<edm::PSimHitContainer> simHits; 
  event.getByToken(simhitToken_, simHits);    

  edm::Handle<GEMDigiCollection> digis;
  event.getByToken(gemDigiToken_, digis);
   
  edm::Handle< edm::DetSetVector<StripDigiSimLink> > thelinkDigis;
  event.getByToken(gemDigiSimLinkToken_, thelinkDigis);

  GEMDigiCollection::DigiRangeIterator detUnitIt;
  for (detUnitIt = digis->begin();	detUnitIt != digis->end(); ++detUnitIt)
  {
    const GEMDetId& id = (*detUnitIt).first;
    const GEMEtaPartition* roll = pDD->etaPartition(id);

    //     if(id.rawId() != 637567293) continue;

    // GEMDetId print-out
    LogDebug("GEMDigiReader")<<"--------------"<<endl;
    LogDebug("GEMDigiReader")<<"id: "<<id.rawId()<<" number of strips "<<roll->nstrips()<<endl;

    // Loop over the digis of this DetUnit
    const GEMDigiCollection::Range& range = (*detUnitIt).second;
    for (GEMDigiCollection::const_iterator digiIt = range.first; digiIt!=range.second; ++digiIt)
    {
      LogDebug("GEMDigiReader")<<" digi "<<*digiIt<<endl;
      if (digiIt->strip() < 1 || digiIt->strip() > roll->nstrips() )
      {
        LogDebug("GEMDigiReader") <<" XXXXXXXXXXXXX Problemt with "<<id<<"  a digi has strip# = "<<digiIt->strip()<<endl;
      } 
      for(const auto& simHit: *simHits)
      {
        GEMDetId rpcId(simHit.detUnitId());
        if (rpcId == id && abs(simHit.particleType()) == 13)
        {
          LogDebug("GEMDigiReader")<<"entry: "<< simHit.entryPoint()<<endl
				   <<"exit: "<< simHit.exitPoint()<<endl
				   <<"TOF: "<< simHit.timeOfFlight()<<endl;
        }
      }
    }// for digis in layer
  }// for layers

  for (edm::DetSetVector<StripDigiSimLink>::const_iterator itlink = thelinkDigis->begin(); itlink != thelinkDigis->end(); itlink++)
  {
    for(edm::DetSet<StripDigiSimLink>::const_iterator link_iter=itlink->data.begin();link_iter != itlink->data.end();++link_iter)
    {
      int detid = itlink->detId();
      int ev = link_iter->eventId().event();
      float frac =  link_iter->fraction();
      int strip = link_iter->channel();
      int trkid = link_iter->SimTrackId();
      int bx = link_iter->eventId().bunchCrossing();
      LogDebug("GEMDigiReader")<<"DetUnit: "<<GEMDetId(detid)<<"  Event ID: "<<ev<<"  trkId: "<<trkid<<"  Strip: "<<strip<<"  Bx: "<<bx<<"  frac: "<<frac<<endl;
    }
  }

  LogDebug("GEMDigiReader")<<"--------------"<<endl;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GEMDigiReader);

