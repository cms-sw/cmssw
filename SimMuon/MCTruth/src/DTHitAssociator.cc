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
#include <string>
#include <sstream>

using namespace std;

// Constructor
DTHitAssociator::DTHitAssociator(const edm::ParameterSet& conf, 
				 edm::ConsumesCollector && iC) :
  DTsimhitsTag(conf.getParameter<edm::InputTag>("DTsimhitsTag")),
  DTsimhitsXFTag(conf.getParameter<edm::InputTag>("DTsimhitsXFTag")),
  DTdigiTag(conf.getParameter<edm::InputTag>("DTdigiTag")),
  DTdigisimlinkTag(conf.getParameter<edm::InputTag>("DTdigisimlinkTag")),
  DTrechitTag(conf.getParameter<edm::InputTag>("DTrechitTag")),

  // nice printout of DT hits
  dumpDT(conf.getParameter<bool>("dumpDT")),
  // CrossingFrame used or not ?
  crossingframe(conf.getParameter<bool>("crossingframe")),
  // Event contain the DTDigiSimLink collection ?
  links_exist(conf.getParameter<bool>("links_exist")),
  // associatorByWire links to a RecHit all the "valid" SimHits on the same DT wire
  associatorByWire(conf.getParameter<bool>("associatorByWire")),

  printRtS(true)
{

  if ( crossingframe) {
    iC.consumes<CrossingFrame<PSimHit> >(DTsimhitsXFTag);
  }
  else if (!DTsimhitsTag.label().empty()) {
    iC.consumes<edm::PSimHitContainer>(DTsimhitsTag);
  }
  iC.consumes<DTDigiCollection>(DTdigiTag);
  iC.consumes<DTDigiSimLinkCollection>(DTdigisimlinkTag);

  if ( dumpDT && printRtS ) {
    iC.consumes<DTRecHitCollection>(DTrechitTag);
  }

}


DTHitAssociator::DTHitAssociator(const edm::Event& iEvent, const edm::EventSetup& iSetup, const edm::ParameterSet& conf, bool printRtS):
  // input collection labels
  DTsimhitsTag(conf.getParameter<edm::InputTag>("DTsimhitsTag")),
  DTsimhitsXFTag(conf.getParameter<edm::InputTag>("DTsimhitsXFTag")),
  DTdigiTag(conf.getParameter<edm::InputTag>("DTdigiTag")),
  DTdigisimlinkTag(conf.getParameter<edm::InputTag>("DTdigisimlinkTag")),
  DTrechitTag(conf.getParameter<edm::InputTag>("DTrechitTag")),

  // nice printout of DT hits
  dumpDT(conf.getParameter<bool>("dumpDT")),
  // CrossingFrame used or not ?
  crossingframe(conf.getParameter<bool>("crossingframe")),
  // Event contain the DTDigiSimLink collection ?
  links_exist(conf.getParameter<bool>("links_exist")),
  // associatorByWire links to a RecHit all the "valid" SimHits on the same DT wire
  associatorByWire(conf.getParameter<bool>("associatorByWire")),

  printRtS(true)
  
{  
  initEvent(iEvent,iSetup);
}

void DTHitAssociator::initEvent(const edm::Event &iEvent, const edm::EventSetup& iSetup) {

  LogTrace("DTHitAssociator") <<"DTHitAssociator constructor: dumpDT = "<<dumpDT
                              <<", crossingframe = "<<crossingframe<<", links_exist = "<<links_exist
                              <<", associatorByWire = "<<associatorByWire;
  
  if (!links_exist && !associatorByWire) {
    edm::LogWarning("DTHitAssociator")<<"*** WARNING in DTHitAssociator::DTHitAssociator: associatorByWire reset to TRUE !"
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
    LogTrace("DTHitAssociator") <<"getting CrossingFrame<PSimHit> collection - "<<DTsimhitsXFTag;
    iEvent.getByLabel(DTsimhitsXFTag,xFrame);
    auto_ptr<MixCollection<PSimHit> > 
      DTsimhits( new MixCollection<PSimHit>(xFrame.product()) );
    LogTrace("DTHitAssociator") <<"... size = "<<DTsimhits->size();
    MixCollection<PSimHit>::MixItr isimhit;
    for (isimhit=DTsimhits->begin(); isimhit!= DTsimhits->end(); isimhit++) {
      DTWireId wireid((*isimhit).detUnitId());
      takeHit = SimHitOK(muonGeom, *isimhit);
      mapOfSimHit[wireid].push_back(make_pair(*isimhit,takeHit));
    }
  }
  else if (!DTsimhitsTag.label().empty()) {
    edm::Handle<edm::PSimHitContainer> DTsimhits;
    LogTrace("DTHitAssociator") <<"getting PSimHit collection - "<<DTsimhitsTag;
    iEvent.getByLabel(DTsimhitsTag,DTsimhits);    
    LogTrace("DTHitAssociator") <<"... size = "<<DTsimhits->size();
    edm::PSimHitContainer::const_iterator isimhit;
    for (isimhit=DTsimhits->begin(); isimhit!= DTsimhits->end(); isimhit++) {
      DTWireId wireid((*isimhit).detUnitId());
      takeHit = SimHitOK(muonGeom, *isimhit);
      mapOfSimHit[wireid].push_back(make_pair(*isimhit,takeHit));
    }    
  }

  // Get the DT Digi collection from the event
  mapOfDigi.clear();
  edm::Handle<DTDigiCollection> digis;
  LogTrace("DTHitAssociator") <<"getting DTDigi collection - "<<DTdigiTag;
  iEvent.getByLabel(DTdigiTag,digis);
  
  if (digis.isValid()) {
    // Map DTDigi by DTWireId
    for (DTDigiCollection::DigiRangeIterator detUnit=digis->begin(); detUnit !=digis->end(); ++detUnit) {
      const DTLayerId& layerid = (*detUnit).first;
      const DTDigiCollection::Range& range = (*detUnit).second;
      
      DTDigiCollection::const_iterator digi;
      for (digi = range.first; digi != range.second; ++digi){
	DTWireId wireid(layerid,(*digi).wire());
	mapOfDigi[wireid].push_back(*digi);
      }
    }
  } else {
    LogTrace("DTHitAssociator") <<"... NO DTDigi collection found !";
  }
  
  mapOfLinks.clear();
  if (links_exist) {
  // Get the DT DigiSimLink collection from the event and map DTDigiSimLink by DTWireId
    edm::Handle<DTDigiSimLinkCollection> digisimlinks;
    LogTrace("DTHitAssociator") <<"getting DTDigiSimLink collection - "<<DTdigisimlinkTag;
    iEvent.getByLabel(DTdigisimlinkTag,digisimlinks);
    
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

  if(dumpDT && printRtS) {
    
    // Get the DT rechits from the event
    edm::Handle<DTRecHitCollection> DTrechits; 
    LogTrace("DTHitAssociator") <<"getting DTRecHit1DPair collection - "<<DTrechitTag;
    iEvent.getByLabel(DTrechitTag,DTrechits);
    LogTrace("DTHitAssociator") <<"... size = "<<DTrechits->size();
    
    // map DTRecHit1DPair by DTWireId
    mapOfRecHit.clear();
    DTRecHitCollection::const_iterator rechit;
    for(rechit=DTrechits->begin(); rechit!=DTrechits->end(); ++rechit) {
      DTWireId wireid = (*rechit).wireId();
      mapOfRecHit[wireid].push_back(*rechit);
    }
    
    if (mapOfSimHit.end() != mapOfSimHit.begin()) {
      edm::LogVerbatim("DTHitAssociator")<<"\n *** Dump DT PSimHit's ***";
      
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
	  edm::LogVerbatim("DTHitAssociator")
	    <<"PSimHit "<<ihit <<", wire "<<wireid //<<", detID = "<<hit.detUnitId()
	    <<", SimTrack Id:"<<hit.trackId()<<"/Evt:(" <<hit.eventId().event()<<","<<hit.eventId().bunchCrossing()<<") "
	    <<", pdg = "<<hit.particleType()<<", procs = "<<hit.processType();
	}	
      } 
    } else {
      edm::LogVerbatim("DTHitAssociator")<<"\n *** There are NO DT PSimHit's ***";
    }

    if(mapOfDigi.end() != mapOfDigi.begin()) {
      
      int jwire = 0;
      int ihit = 0;
      
      for(DigiMap::const_iterator mapIT=mapOfDigi.begin(); mapIT!=mapOfDigi.end(); ++mapIT , jwire++) {
	edm::LogVerbatim("DTHitAssociator")<<"\n *** Dump DT digis ***";

	DTWireId wireid = (*mapIT).first;
	for (vector<DTDigi>::const_iterator hitIT = mapOfDigi[wireid].begin(); 
	     hitIT != mapOfDigi[wireid].end(); 
	     hitIT++ , ihit++) {
	  edm::LogVerbatim("DTHitAssociator")
	    <<"DTDigi "<<ihit<<", wire "<<wireid<<", number = "<<hitIT->number()<<", TDC counts = "<<hitIT->countsTDC();
	}	
      }
    } else {
      LogTrace("DTHitAssociator")<<"\n *** There are NO DTDigi's ***";
    }
    
    if (mapOfRecHit.end() != mapOfRecHit.begin()) 
      edm::LogVerbatim("DTHitAssociator")<<"\n *** Analyze DTRecHitCollection by DTWireId ***";
    
    int iwire = 0;
    for(RecHitMap::const_iterator mapIT=mapOfRecHit.begin(); 
	mapIT!=mapOfRecHit.end(); 
	++mapIT , iwire++) {
      
      DTWireId wireid = (*mapIT).first;
      edm::LogVerbatim("DTHitAssociator")<<"\n==================================================================="; 
      edm::LogVerbatim("DTHitAssociator")<<"wire index = "<<iwire<<"  *** DTWireId = "<<" ("<<wireid<<")";
      
      if(mapOfSimHit.find(wireid) != mapOfSimHit.end()) {
	edm::LogVerbatim("DTHitAssociator")<<"\n"<<mapOfSimHit[wireid].size()<<" SimHits (PSimHit):";
	
	for (vector<PSimHit_withFlag>::const_iterator hitIT = mapOfSimHit[wireid].begin(); 
	     hitIT != mapOfSimHit[wireid].end(); 
	     ++hitIT) {
	  stringstream tId;
	  tId << (hitIT->first).trackId();
	  string simhitlog = "\t SimTrack Id = "+tId.str();
	  if (hitIT->second) simhitlog = simhitlog + "\t -VALID HIT-";
	  else simhitlog = simhitlog + "\t -not valid hit-";
	  edm::LogVerbatim("DTHitAssociator")<<simhitlog;
	}
      }
      
      if(mapOfLinks.find(wireid) != mapOfLinks.end()) {
	edm::LogVerbatim("DTHitAssociator")<<"\n"<<mapOfLinks[wireid].size()<<" Links (DTDigiSimLink):";
	
	for (vector<DTDigiSimLink>::const_iterator hitIT = mapOfLinks[wireid].begin(); 
	     hitIT != mapOfLinks[wireid].end(); 
	     ++hitIT) {
	  edm::LogVerbatim("DTHitAssociator")
	    <<"\t digi number = "<<hitIT->number()<<", time = "<<hitIT->time()<<", SimTrackId = "<<hitIT->SimTrackId();
	}
      }
      
      if(mapOfDigi.find(wireid) != mapOfDigi.end()) {
	edm::LogVerbatim("DTHitAssociator")<<"\n"<<mapOfDigi[wireid].size()<<" Digis (DTDigi):";
	for (vector<DTDigi>::const_iterator hitIT = mapOfDigi[wireid].begin(); 
	     hitIT != mapOfDigi[wireid].end(); 
	     ++hitIT) {
	  edm::LogVerbatim("DTHitAssociator")<<"\t digi number = "<<hitIT->number()<<", time = "<<hitIT->time();
	}
      }
      
      edm::LogVerbatim("DTHitAssociator")<<"\n"<<(*mapIT).second.size()<<" RecHits (DTRecHit1DPair):";    
      for(vector<DTRecHit1DPair>::const_iterator vIT =(*mapIT).second.begin(); 
	  vIT !=(*mapIT).second.end(); 
	  ++vIT) {
	edm::LogVerbatim("DTHitAssociator")<<"\t digi time = "<<vIT->digiTime();
      }
    }
  }
}
// end of constructor

std::vector<DTHitAssociator::SimHitIdpr> DTHitAssociator::associateHitId(const TrackingRecHit & hit) const {
  
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

std::vector<DTHitAssociator::SimHitIdpr> DTHitAssociator::associateDTHitId(const DTRecHit1D * dtrechit) const {
  
  std::vector<SimHitIdpr> matched; 

  DTWireId wireid = dtrechit->wireId();
  
  if(associatorByWire) {
    // matching based on DTWireId : take only "valid" SimHits on that wire

    auto found = mapOfSimHit.find(wireid);
    if(found != mapOfSimHit.end()) {	
      for (vector<PSimHit_withFlag>::const_iterator hitIT = found->second.begin(); 
	   hitIT != found->second.end(); 
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
    
    auto found = mapOfLinks.find(wireid);

    if (found != mapOfLinks.end()) {
      // DTDigiSimLink::time() is set equal to DTDigi::time() only for the DTDigiSimLink corresponding to the digitizied PSimHit
      // other DTDigiSimLinks associated to the same DTDigi are identified by having the same DTDigiSimLink::number()
      
      // first find the associated digi Number
      for (vector<DTDigiSimLink>::const_iterator linkIT = found->second.begin(); 
	   linkIT != found->second.end(); 
	   ++linkIT ) {
	float digitime = linkIT->time();
	if (fabs(digitime-theTime)<0.1) {
          theNumber = linkIT->number();
	}	
      }
      
      // then get all the DTDigiSimLinks with that digi Number (corresponding to valid SimHits  
      //  within a time window of the order of the time resolution, specified in the DTDigitizer)
      for (vector<DTDigiSimLink>::const_iterator linkIT = found->second.begin(); 
	   linkIT != found->second.end(); 
	   ++linkIT ) {
	
        int digiNr = linkIT->number();
	if (digiNr == theNumber) {
	  SimHitIdpr currentId(linkIT->SimTrackId(), linkIT->eventId());
	  matched.push_back(currentId);
	}
      }

    } else {
      edm::LogError("DTHitAssociator")<<"*** ERROR in DTHitAssociator::associateDTHitId - DTRecHit1D: "
				      <<*dtrechit<<" has no associated DTDigiSimLink !"<<endl;
      return matched; 
    }
  }
  
  return matched;
}

std::vector<PSimHit> DTHitAssociator::associateHit(const TrackingRecHit & hit) const {
  
  std::vector<PSimHit> simhits;  
  std::vector<SimHitIdpr> simtrackids;

  const TrackingRecHit * hitp = &hit;
  const DTRecHit1D * dtrechit = dynamic_cast<const DTRecHit1D *>(hitp);

  if (dtrechit) {
    DTWireId wireid = dtrechit->wireId();
    
    if (associatorByWire) {
    // matching based on DTWireId : take only "valid" SimHits on that wire

      auto found = mapOfSimHit.find(wireid);
      if(found != mapOfSimHit.end()) {	
	for (vector<PSimHit_withFlag>::const_iterator hitIT = found->second.begin(); 
	     hitIT != found->second.end(); 
	     ++hitIT) {
	  
	  bool valid_hit = hitIT->second;
	  if (valid_hit) simhits.push_back(hitIT->first);
	}
      }
    }
    else {
      // matching based on DTDigiSimLink

      simtrackids = associateDTHitId(dtrechit);

      for (vector<SimHitIdpr>::const_iterator idIT =  simtrackids.begin(); idIT != simtrackids.end(); ++idIT) {
	uint32_t trId = idIT->first;
	EncodedEventId evId = idIT->second;
	
        auto found = mapOfSimHit.find(wireid);
	if(found != mapOfSimHit.end()) {	
	  for (vector<PSimHit_withFlag>::const_iterator hitIT = found->second.begin(); 
	       hitIT != found->second.end(); 
	       ++hitIT) {
	    
	    if (hitIT->first.trackId() == trId  && 
		hitIT->first.eventId() == evId) 
	      simhits.push_back(hitIT->first);	    
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
