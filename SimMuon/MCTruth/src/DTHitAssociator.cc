#include "SimMuon/MCTruth/interface/DTHitAssociator.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

// Constructor
DTHitAssociator::DTHitAssociator(const edm::Event& iEvent, const edm::EventSetup& iSetup, const edm::ParameterSet& conf) {

  // nice printout of DT hits
  dumpDT=conf.getParameter<bool>("dumpDT");
  // CrossingFrame used or not ?
  crossingframe=conf.getParameter<bool>("crossingframe");
  // Event contain the DTDigiSimLink collection ?
  links_exist=conf.getParameter<bool>("links_exist");
  // associatorByWire links to a RecHit all the "valid" SimHits on the same DT wire
  associatorByWire=conf.getParameter<bool>("associatorByWire");
  
  LogTrace("DTHitAssociator") <<"DTHitAssociator constructor: dumpDT = "<<dumpDT
			      <<", crossingframe = "<<crossingframe<<", links_exist = "<<links_exist
			      <<", associatorByWire = "<<associatorByWire;

  if (!links_exist && !associatorByWire) {
    edm::LogWarning("DTHitAssociator")<<"DTHitAssociator: WARNING: associatorByWire reset to TRUE !"
				      <<"    \t (missing DTDigiSimLinkCollection ?)";
    associatorByWire = true;
  }
  
  // need DT Geometry to discard hits for which the drift time parametrisation is not applicable
  edm::ESHandle<DTGeometry> muonGeom;
  iSetup.get<MuonGeometryRecord>().get(muonGeom);
  
  // Get the DT SimHits from the event and map PSimHit by DTWireId
  mapOfSimHit.clear();
  bool takeHit(true);
 
  if (crossingframe) {
    edm::Handle<CrossingFrame<PSimHit> > xFrame;
    LogTrace("DTHitAssociator") <<"getting CrossingFrame<PSimHit> collection with label=mix, instance=MuonDTHits ";
    iEvent.getByLabel("mix","MuonDTHits",xFrame);
    auto_ptr<MixCollection<PSimHit> > 
      DTsimhits( new MixCollection<PSimHit>(xFrame.product()) );
    LogTrace("DTHitAssociator") <<"CrossingFrame<PSimHit> -MuonDTHits- collection has size = "<<DTsimhits->size();
    MixCollection<PSimHit>::MixItr isimhit;
    for (isimhit=DTsimhits->begin(); isimhit!= DTsimhits->end(); isimhit++) {
      DTWireId wireid((*isimhit).detUnitId());
      takeHit = SimHitOK(muonGeom, *isimhit);
      mapOfSimHit[wireid].push_back(make_pair(*isimhit,takeHit));
    }
  }
  else {
    edm::Handle<edm::PSimHitContainer> DTsimhits;
    LogTrace("DTHitAssociator") <<"getting PSimHit collection with label=g4SimHits, instance=MuonDTHits ";
    iEvent.getByLabel("g4SimHits","MuonDTHits",DTsimhits);    
    LogTrace("DTHitAssociator") <<"PSimHit collection -MuonDTHits- has size = "<<DTsimhits->size();
    edm::PSimHitContainer::const_iterator isimhit;
    for (isimhit=DTsimhits->begin(); isimhit!= DTsimhits->end(); isimhit++) {
      DTWireId wireid((*isimhit).detUnitId());
      takeHit = SimHitOK(muonGeom, *isimhit);
      mapOfSimHit[wireid].push_back(make_pair(*isimhit,takeHit));
    }    
  }

  // Get the DT Digi collection from the event
  edm::Handle<DTDigiCollection> digis;
  LogTrace("DTHitAssociator") <<"getting DTDigi collection with label=muonDTDigis ";
  //  iEvent.getByLabel("muonDTDigis","MuonDTDigis",digis);
  iEvent.getByLabel("muonDTDigis",digis);

  // Map DTDigi by DTWireId
  mapOfDigi.clear();
  for (DTDigiCollection::DigiRangeIterator detUnit=digis->begin(); detUnit !=digis->end(); ++detUnit) {
    const DTLayerId& layerid = (*detUnit).first;
    const DTDigiCollection::Range& range = (*detUnit).second;

    DTDigiCollection::const_iterator digi;
    for (digi = range.first; digi != range.second; ++digi){
      DTWireId wireid(layerid,(*digi).wire());
      mapOfDigi[wireid].push_back(*digi);
    }
  }

  mapOfLinks.clear();
  if (links_exist) {
  // Get the DT DigiSimLink collection from the event and map DTDigiSimLink by DTWireId
    edm::Handle<DTDigiSimLinkCollection> digisimlinks;
    LogTrace("DTHitAssociator") <<"getting DTDigiSimLink collection with label=muonDTDigis ";
    //  iEvent.getByLabel("muonDTDigis","MuonDTDigiSimLinks",digisimlinks);
    iEvent.getByLabel("muonDTDigis",digisimlinks);
    
    for (DTDigiSimLinkCollection::DigiRangeIterator detUnit=digisimlinks->begin(); 
	 detUnit !=digisimlinks->end(); 
	 ++detUnit) {
      const DTLayerId& layerid = (*detUnit).first;
      const DTDigiSimLinkCollection::Range& range = (*detUnit).second;
      
      DTDigiSimLinkCollection::const_iterator link;
      for (link=range.first; link!=range.second; ++link){
	DTWireId wireid(layerid,(*link).wire());
	mapOfLinks[wireid].push_back(*link);
      }
    }
  }

  // Get the DT rechits from the event
  edm::Handle<DTRecHitCollection> DTrechits; 
  LogTrace("DTHitAssociator") <<"getting DTRecHit1DPair collection with label=dt1DRecHits ";
  iEvent.getByLabel("dt1DRecHits",DTrechits);
  LogTrace("DTHitAssociator") <<"DTRecHit1DPair collection has size = "<<DTrechits->size();
  
  // map DTRecHit1DPair by DTWireId
  mapOfRecHit.clear();
  DTRecHitCollection::const_iterator rechit;
  for(rechit=DTrechits->begin(); rechit!=DTrechits->end(); ++rechit) {
    DTWireId wireid = (*rechit).wireId();
    mapOfRecHit[wireid].push_back(*rechit);
  }
  
  if(dumpDT) {
    
    if (mapOfSimHit.end() != mapOfSimHit.begin())
      cout<<"\n *** Dump DT PSimHit's ***"<<endl;
    
    int jwire = 0;
    int ihit = 0;
    
    for(SimHitMap::const_iterator mapIT=mapOfSimHit.begin();
	mapIT!=mapOfSimHit.end();
	++mapIT , jwire++) {
      
      DTWireId wireid = (*mapIT).first;
      for (vector<PSimHit_withFlag>::const_iterator hitIT = mapOfSimHit[wireid].begin(); 
	   hitIT != mapOfSimHit[wireid].end(); 
	   hitIT++ , ihit++) {
	PSimHit hit = hitIT->first;
	cout<<"PSimHit "<<ihit
	  //<<", detID = "<<hit.detUnitId()
	    <<", wire "<<wireid
	    <<", SimTrack Id:"<<hit.trackId()
	    <<"/Evt:(" <<hit.eventId().event()<<","<<hit.eventId().bunchCrossing()<<") "
	    <<", pdg = "<<hit.particleType()<<", procs = "<<hit.processType()<<endl;
      }
      
    }

    if (mapOfRecHit.end() != mapOfRecHit.begin()) 
      cout<<"\n *** Analyze DTRecHitCollection by DTWireId ***"<<endl;
    
    int iwire = 0;
    for(RecHitMap::const_iterator mapIT=mapOfRecHit.begin(); 
	mapIT!=mapOfRecHit.end(); 
	++mapIT , iwire++) {
      
      DTWireId wireid = (*mapIT).first;
      cout<<"\n==================================================================="<<endl; 
      cout<<"wire index = "<<iwire<<"  *** DTWireId = "<<" ("<<wireid<<")"<<endl;
      
      if(mapOfSimHit.find(wireid) != mapOfSimHit.end()) {
	cout<<endl<<mapOfSimHit[wireid].size()<<" SimHits (PSimHit):"<<endl;
	
	for (vector<PSimHit_withFlag>::const_iterator hitIT = mapOfSimHit[wireid].begin(); 
	     hitIT != mapOfSimHit[wireid].end(); 
	     ++hitIT) {
	  cout<<"\t SimTrack Id = "<<(hitIT->first).trackId();
	  if (hitIT->second) cout<<"\t -VALID HIT-"<<endl;
	  else cout<<"\t -not valid hit-"<<endl;
	}
      }
      
      if(mapOfLinks.find(wireid) != mapOfLinks.end()) {
	cout<<endl<<mapOfLinks[wireid].size()<<" Links (DTDigiSimLink):"<<endl;
	
	for (vector<DTDigiSimLink>::const_iterator hitIT = mapOfLinks[wireid].begin(); 
	     hitIT != mapOfLinks[wireid].end(); 
	     ++hitIT) {
	  cout<<"\t digi number = "<<hitIT->number()<<", time = "<<hitIT->time()
	      <<", SimTrackId = "<<hitIT->SimTrackId()<<endl;
	}
      }
      
      if(mapOfDigi.find(wireid) != mapOfDigi.end()) {
	cout<<endl<<mapOfDigi[wireid].size()<<" Digis (DTDigi):"<<endl;
	
	for (vector<DTDigi>::const_iterator hitIT = mapOfDigi[wireid].begin(); 
	     hitIT != mapOfDigi[wireid].end(); 
	     ++hitIT) {
	  cout<<"\t digi number = "<<hitIT->number()<<", time = "<<hitIT->time()<<endl;
	}
      }
      
      cout<<endl<<(*mapIT).second.size()<<" RecHits (DTRecHit1DPair):"<<endl;
      
      for(vector<DTRecHit1DPair>::const_iterator vIT =(*mapIT).second.begin(); 
	  vIT !=(*mapIT).second.end(); 
	  ++vIT) {
	cout<<"\t digi time = "<<vIT->digiTime()<<endl;
      }
    }
  }
}
// end of constructor

std::vector<DTHitAssociator::SimHitIdpr> DTHitAssociator::associateHitId(const TrackingRecHit & hit) {
  
  std::vector<SimHitIdpr> simtrackids;
  const TrackingRecHit * hitp = &hit;
  const DTRecHit1D * dtrechit = dynamic_cast<const DTRecHit1D *>(hitp);
  
  if (dtrechit) {
    simtrackids = associateDTHitId(dtrechit);
  } else {
    edm::LogWarning("DTHitAssociator")<<"*** WARNING in DTHitAssociator::associateHitId, null dynamic_cast !";
  }
  return simtrackids;
}

std::vector<DTHitAssociator::SimHitIdpr> DTHitAssociator::associateDTHitId(const DTRecHit1D * dtrechit) {
  
  std::vector<SimHitIdpr> matched; 

  DTWireId wireid = dtrechit->wireId();
  
  if(associatorByWire) {
    // matching based on DTWireId : take only "valid" SimHits on that wire

    if(mapOfSimHit.find(wireid) != mapOfSimHit.end()) {	
      for (vector<PSimHit_withFlag>::const_iterator hitIT = mapOfSimHit[wireid].begin(); 
	   hitIT != mapOfSimHit[wireid].end(); 
	   ++hitIT) {
	
	bool valid_hit = hitIT->second;
	PSimHit this_hit = hitIT->first;

	if (valid_hit) {
	  SimHitIdpr currentId(this_hit.trackId(), this_hit.eventId());
	  matched.push_back(currentId);
	}
      }
    }
  }

  else {
    // matching based on DTDigiSimLink
    
    float theTime = dtrechit->digiTime();
    int theNumber(-1);

    if (mapOfLinks.find(wireid) != mapOfLinks.end()) {
      // first find the associated digi Number
      for (vector<DTDigiSimLink>::const_iterator linkIT = mapOfLinks[wireid].begin(); 
	   linkIT != mapOfLinks[wireid].end(); 
	   ++linkIT ) {
	
	float digitime = linkIT->time();
	if (fabs(digitime-theTime)<0.1) {
          theNumber = linkIT->number();
	}	
      }
      
      // then get all the DTDigiSimLinks with that digi Number (corresponding to valid SimHits  
      //  within a time window of the order of the time resolution, specified in the DTDigitizer)
      for (vector<DTDigiSimLink>::const_iterator linkIT = mapOfLinks[wireid].begin(); 
	   linkIT != mapOfLinks[wireid].end(); 
	   ++linkIT ) {
	
        int digiNr = linkIT->number();
	if (digiNr == theNumber) {
	  SimHitIdpr currentId(linkIT->SimTrackId(), linkIT->eventId());
	  matched.push_back(currentId);
	}
      }

    } else {
      edm::LogError("DTHitAssociator")<<"ERROR in DTHitAssociator::associateDTHitId - DTRecHit1D: "
				      <<*dtrechit<<" has no associated DTDigiSimLink !"<<endl;
      return matched; 
    }
  }
  
  return matched;
}

std::vector<const PSimHit *> DTHitAssociator::associateHit(const TrackingRecHit & hit) {
  
  std::vector<const PSimHit *> simhits;  
  std::vector<SimHitIdpr> simtrackids;

  const TrackingRecHit * hitp = &hit;
  const DTRecHit1D * dtrechit = dynamic_cast<const DTRecHit1D *>(hitp);

  if (dtrechit) {
    //    simtrackids = associateDTHitId(dtrechit);
    DTWireId wireid = dtrechit->wireId();
    
    if (associatorByWire) {
    // matching based on DTWireId : take only "valid" SimHits on that wire

      if(mapOfSimHit.find(wireid) != mapOfSimHit.end()) {	
	for (vector<PSimHit_withFlag>::const_iterator hitIT = mapOfSimHit[wireid].begin(); 
	     hitIT != mapOfSimHit[wireid].end(); 
	     ++hitIT) {
	  
	  bool valid_hit = hitIT->second;
	  if (valid_hit) simhits.push_back( &(hitIT->first) );
	}
      }
    }
    else {
      // matching based on DTDigiSimLink

      simtrackids = associateDTHitId(dtrechit);

      for (vector<SimHitIdpr>::const_iterator idIT =  simtrackids.begin(); idIT != simtrackids.end(); ++idIT) {
	uint32_t trId = idIT->first;
	EncodedEventId evId = idIT->second;
	
	if(mapOfSimHit.find(wireid) != mapOfSimHit.end()) {	
	  for (vector<PSimHit_withFlag>::const_iterator hitIT = mapOfSimHit[wireid].begin(); 
	       hitIT != mapOfSimHit[wireid].end(); 
	       ++hitIT) {
	    
	    if (hitIT->first.trackId() == trId  && 
		hitIT->first.eventId() == evId) 
	      simhits.push_back( &(hitIT->first) );	    
	  }
	}	
      }
    }

  } else {
    edm::LogWarning("DTHitAssociator")<<"*** WARNING in DTHitAssociator::associateHit, null dynamic_cast !";
  }
  return simhits;
}

bool DTHitAssociator::SimHitOK(const edm::ESHandle<DTGeometry> & muongeom, 
			       const PSimHit & simhit) {
  bool result = true;
  
  DTWireId wireid(simhit.detUnitId());
  const DTLayer* dtlayer = muongeom->layer(wireid.layerId()); 
  LocalPoint entryP = simhit.entryPoint();
  LocalPoint exitP = simhit.exitPoint();
  const DTTopology &topo = dtlayer->specificTopology();
  float xwire = topo.wirePosition(wireid.wire()); 
  float xEntry = entryP.x() - xwire;
  float xExit  = exitP.x() - xwire;
  DTTopology::Side entrySide = topo.onWhichBorder(xEntry,entryP.y(),entryP.z());
  DTTopology::Side exitSide  = topo.onWhichBorder(xExit,exitP.y(),exitP.z());
  
  bool noParametrisation = 
    (( entrySide == DTTopology::none || exitSide == DTTopology::none ) ||
     (entrySide == exitSide) ||
     ((entrySide == DTTopology::xMin && exitSide == DTTopology::xMax) || 
      (entrySide == DTTopology::xMax && exitSide == DTTopology::xMin))   );

  // discard hits where parametrization can not be used
  if (noParametrisation) 
    {
      result = false;
      return result;
    }  
    
  float x;
  LocalPoint hitPos = simhit.localPosition(); 
  
  if(fabs(hitPos.z()) < 0.002) {
    // hit center within 20 um from z=0, no need to extrapolate.
    x = hitPos.x() - xwire;
  } else {
    x = xEntry - (entryP.z()*(xExit-xEntry))/(exitP.z()-entryP.z());
  }
  
  // discard hits where x is out of range of the parametrization (|x|>2.1 cm)
  x *= 10.;  //cm -> mm 
  if (fabs(x) > 21.) 
    result = false;
  
  return result;
}
