#include "SimMuon/MCTruth/interface/CSCHitAssociator.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

CSCHitAssociator::CSCHitAssociator(const edm::Event& event, const edm::EventSetup& setup, const edm::ParameterSet& conf): 
  theDigiSimLinks(0),
  linksTag(conf.getParameter<edm::InputTag>("CSClinksTag"))
{
  initEvent(event,setup);
}

CSCHitAssociator::CSCHitAssociator( const edm::ParameterSet& conf, edm::ConsumesCollector && iC ): 
  theDigiSimLinks(0),
  linksTag(conf.getParameter<edm::InputTag>("CSClinksTag"))
{
  iC.consumes<DigiSimLinks>(linksTag);
}

void CSCHitAssociator::initEvent(const edm::Event& event, const edm::EventSetup& setup) {

  edm::Handle<DigiSimLinks> digiSimLinks;
  LogTrace("CSCHitAssociator") <<"getting CSC Strip DigiSimLink collection - "<<linksTag;
  event.getByLabel(linksTag, digiSimLinks);
  theDigiSimLinks = digiSimLinks.product();

  // get CSC Geometry to use CSCLayer methods
  edm::ESHandle<CSCGeometry> mugeom;
  setup.get<MuonGeometryRecord>().get( mugeom );
  cscgeom = &*mugeom;
}

std::vector<CSCHitAssociator::SimHitIdpr> CSCHitAssociator::associateCSCHitId(const CSCRecHit2D * cscrechit) const {
  std::vector<SimHitIdpr> simtrackids;
  
  unsigned int detId = cscrechit->geographicalId().rawId();
  int nchannels = cscrechit->nStrips();
  const CSCLayerGeometry * laygeom = cscgeom->layer(cscrechit->cscDetId())->geometry();
  
  DigiSimLinks::const_iterator layerLinks = theDigiSimLinks->find(detId);    
  
  if (layerLinks != theDigiSimLinks->end()) {
    
    for(int idigi = 0; idigi < nchannels; ++idigi) {
      // strip and readout channel numbers may differ in ME1/1A
      int istrip = cscrechit->channels(idigi);
      int channel = laygeom->channel(istrip);
      
      for (LayerLinks::const_iterator link=layerLinks->begin(); link!=layerLinks->end(); ++link) {
	int ch = static_cast<int>(link->channel());
	if (ch == channel) {
	  SimHitIdpr currentId(link->SimTrackId(), link->eventId());
	  if (find(simtrackids.begin(), simtrackids.end(), currentId) == simtrackids.end())
	    simtrackids.push_back(currentId);
	}
      }
    }
    
  } else edm::LogWarning("CSCHitAssociator")
    <<"*** WARNING in CSCHitAssociator::associateCSCHitId - CSC layer "<<detId<<" has no DigiSimLinks !"<<std::endl;   
  
  return simtrackids;
}


std::vector<CSCHitAssociator::SimHitIdpr> CSCHitAssociator::associateHitId(const TrackingRecHit & hit) const
{
  std::vector<SimHitIdpr> simtrackids;
  
  const TrackingRecHit * hitp = &hit;
  const CSCRecHit2D * cscrechit = dynamic_cast<const CSCRecHit2D *>(hitp);

  if (cscrechit) {
    
    unsigned int detId = cscrechit->geographicalId().rawId();
    int nchannels = cscrechit->nStrips();
    const CSCLayerGeometry * laygeom = cscgeom->layer(cscrechit->cscDetId())->geometry();

    DigiSimLinks::const_iterator layerLinks = theDigiSimLinks->find(detId);    

    if (layerLinks != theDigiSimLinks->end()) {
      
      for(int idigi = 0; idigi < nchannels; ++idigi) {
	// strip and readout channel numbers may differ in ME1/1A
	int istrip = cscrechit->channels(idigi);
	int channel = laygeom->channel(istrip);
	
	for (LayerLinks::const_iterator link=layerLinks->begin(); link!=layerLinks->end(); ++link) {
	  int ch = static_cast<int>(link->channel());
	  if (ch == channel) {
	    SimHitIdpr currentId(link->SimTrackId(), link->eventId());
	    if (find(simtrackids.begin(), simtrackids.end(), currentId) == simtrackids.end())
	      simtrackids.push_back(currentId);
	  }
	}
      }
      
    } else edm::LogWarning("CSCHitAssociator")
      <<"*** WARNING in CSCHitAssociator::associateHitId - CSC layer "<<detId<<" has no DigiSimLinks !"<<std::endl;   
    
  } else edm::LogWarning("CSCHitAssociator")<<"*** WARNING in CSCHitAssociator::associateHitId, null dynamic_cast !";
  
  return simtrackids;
}
    

