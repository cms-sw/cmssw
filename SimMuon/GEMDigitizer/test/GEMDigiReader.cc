#ifndef SimMuon_GEMDigiReader_h
#define SimMuon_GEMDigiReader_h

/** \class GEMDigiReader
 *  Dumps GEM digis 
 *  
 *  $Id:$
 *  \authors: Vadim Khotilovich
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

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

  string label;
};



GEMDigiReader::GEMDigiReader(const edm::ParameterSet& pset)
{
  label = pset.getUntrackedParameter<string>("label", "simMuonGEMDigis");
}


void GEMDigiReader::analyze(const edm::Event & event, const edm::EventSetup& eventSetup)
{
  cout << "--- Run: " << event.id().run() << " Event: " << event.id().event() << endl;

  edm::Handle<GEMDigiCollection> digis;
  event.getByLabel(label, digis);

  edm::Handle<edm::PSimHitContainer> simHits; 
  event.getByLabel("g4SimHits","MuonGEMHits",simHits);    

  edm::ESHandle<GEMGeometry> pDD;
  eventSetup.get<MuonGeometryRecord>().get( pDD );
   
  edm::Handle< edm::DetSetVector<StripDigiSimLink> > thelinkDigis;
  event.getByLabel(label, "GEM", thelinkDigis);

  GEMDigiCollection::DigiRangeIterator detUnitIt;
  for (detUnitIt = digis->begin();	detUnitIt != digis->end(); ++detUnitIt)
  {
    const GEMDetId& id = (*detUnitIt).first;
    const GEMEtaPartition* roll = pDD->etaPartition(id);

    //     if(id.rawId() != 637567293) continue;

    // GEMDetId print-out
    cout<<"--------------"<<endl;
    cout<<"id: "<<id.rawId()<<" number of strips "<<roll->nstrips()<<endl;

    // Loop over the digis of this DetUnit
    const GEMDigiCollection::Range& range = (*detUnitIt).second;
    for (GEMDigiCollection::const_iterator digiIt = range.first; digiIt!=range.second; ++digiIt)
    {
      cout<<" digi "<<*digiIt<<endl;
      if (digiIt->strip() < 1 || digiIt->strip() > roll->nstrips() )
      {
        cout <<" XXXXXXXXXXXXX Problemt with "<<id<<"  a digi has strip# = "<<digiIt->strip()<<endl;
      } 
      for(const auto& simHit: *simHits)
      {
        GEMDetId rpcId(simHit.detUnitId());
        if (rpcId == id && abs(simHit.particleType()) == 13)
        {
          cout<<"entry: "<< simHit.entryPoint()<<endl
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
      cout<<"DetUnit: "<<GEMDetId(detid)<<"  Event ID: "<<ev<<"  trkId: "<<trkid<<"  Strip: "<<strip<<"  Bx: "<<bx<<"  frac: "<<frac<<endl;
    }
  }

  cout<<"--------------"<<endl;
}



#endif
#include <FWCore/Framework/interface/MakerMacros.h>
DEFINE_FWK_MODULE(GEMDigiReader);

