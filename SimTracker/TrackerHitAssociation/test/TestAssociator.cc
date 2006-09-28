// File: TestAssociator.cc
// Author:  P. Azzi
// Creation Date:  PA May 2006 Initial version.
//                 Pixel RecHits added by V.Chiochia - 18/5/06
//
//--------------------------------------------
#include <memory>
#include <string>
#include <iostream>

#include "SimTracker/TrackerHitAssociation/test/TestAssociator.h"

//--- for SimHit
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

//--- for Strip RecHit
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/Common/interface/OwnVector.h"

//--- for Pixel RecHit
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"

//--- for StripDigiSimLink
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"

//--- framework stuff
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//--- for Geometry:
#include "DataFormats/DetId/interface/DetId.h"

#include "Geometry/Vector/interface/LocalPoint.h"
#include "Geometry/Vector/interface/GlobalPoint.h"

#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"

//---------------
// Constructor --
//---------------

using namespace std;
using namespace edm;

  void TestAssociator::analyze(const edm::Event& e, const edm::EventSetup& es) {
    
    using namespace edm;
    bool pixeldebug = true;
    int pixelcounter = 0;
    int stripcounter=0;

    
    // Step A: Get Inputs 
    edm::Handle<SiStripMatchedRecHit2DCollection> rechitsmatched;
    edm::Handle<SiStripRecHit2DCollection> rechitsrphi;
    edm::Handle<SiStripRecHit2DCollection> rechitsstereo;
    edm::Handle<SiPixelRecHitCollection> pixelrechits;
    std::string  rechitProducer = "siStripMatchedRecHits";

    if(doStrip_) {
      e.getByLabel(rechitProducer,"matchedRecHit", rechitsmatched);
      e.getByLabel(rechitProducer,"rphiRecHit", rechitsrphi);
      e.getByLabel(rechitProducer,"stereoRecHit", rechitsstereo);
    }
    if(doPixel_) {
      e.getByLabel("siPixelRecHits",pixelrechits);
    }
    if(!doPixel_ && !doStrip_)  throw edm::Exception(errors::Configuration,"Strip and pixel association disabled");
    
    //first instance tracking geometry
    edm::ESHandle<TrackerGeometry> pDD;
    es.get<TrackerDigiGeometryRecord> ().get (pDD);
    
    // loop over detunits
    for(TrackerGeometry::DetContainer::const_iterator it = pDD->dets().begin(); it != pDD->dets().end(); it++){
      uint32_t myid=((*it)->geographicalId()).rawId();       
      DetId detid = ((*it)->geographicalId());
      
      //construct the associator object
      TrackerHitAssociator  associate(e,conf_);
      
      edm::OwnVector<SiStripRecHit2D> collector; 
      if(myid!=999999999){ //if is valid detector

	if(doPixel_) {
	  
	  SiPixelRecHitCollection::range pixelrechitRange = (pixelrechits.product())->get((detid));
	  SiPixelRecHitCollection::const_iterator pixelrechitRangeIteratorBegin = pixelrechitRange.first;
	  SiPixelRecHitCollection::const_iterator pixelrechitRangeIteratorEnd   = pixelrechitRange.second;
	  SiPixelRecHitCollection::const_iterator pixeliter = pixelrechitRangeIteratorBegin;
	  
	  // Do the pixels
	  for ( ; pixeliter != pixelrechitRangeIteratorEnd; ++pixeliter) {
	    pixelcounter++;
	    if(pixeldebug) {
	      	      cout << pixelcounter <<") Pixel RecHit DetId " << detid.rawId() << " Pos = " << pixeliter->localPosition() << endl;
	    }
	    matched.clear();
	    matched = associate.associateHit(*pixeliter);
	    if(!matched.empty()){
	      cout << " PIX detector =  " << myid << " PIX Rechit = " << pixeliter->localPosition() << endl; 
	      cout << " PIX matched = " << matched.size() << endl;
	    for(vector<PSimHit>::const_iterator m=matched.begin(); m<matched.end(); m++){
	      cout << " PIX hit  ID = " << (*m).trackId() << " PIX Simhit x = " << (*m).localPosition() << endl;
	    }
	    }  
	  }
	}
	
	if(doStrip_) {
	  
	  SiStripRecHit2DCollection::range rechitrphiRange = (rechitsrphi.product())->get((detid));
	  SiStripRecHit2DCollection::const_iterator rechitrphiRangeIteratorBegin = rechitrphiRange.first;
	  SiStripRecHit2DCollection::const_iterator rechitrphiRangeIteratorEnd   = rechitrphiRange.second;
	  SiStripRecHit2DCollection::const_iterator iterrphi=rechitrphiRangeIteratorBegin;
	  
	  SiStripRecHit2DCollection::range rechitsterRange = (rechitsstereo.product())->get((detid));
	  SiStripRecHit2DCollection::const_iterator rechitsterRangeIteratorBegin = rechitsterRange.first;
	  SiStripRecHit2DCollection::const_iterator rechitsterRangeIteratorEnd   = rechitsterRange.second;
	  SiStripRecHit2DCollection::const_iterator iterster=rechitsterRangeIteratorBegin;

	  SiStripMatchedRecHit2DCollection::range rechitmatchRange = (rechitsmatched.product())->get((detid));
	  SiStripMatchedRecHit2DCollection::const_iterator rechitmatchRangeIteratorBegin = rechitmatchRange.first;
	  SiStripMatchedRecHit2DCollection::const_iterator rechitmatchRangeIteratorEnd   = rechitmatchRange.second;
	  SiStripMatchedRecHit2DCollection::const_iterator itermatch=rechitmatchRangeIteratorBegin;
	  
	  // Do the strips
	  for(iterrphi=rechitrphiRangeIteratorBegin;iterrphi!=rechitrphiRangeIteratorEnd;++iterrphi){//loop on the rechit
	    SiStripRecHit2D const rechit=*iterrphi;
	    int i=0;
	    stripcounter++;
	    cout << stripcounter <<") Strip RecHit DetId " << detid.rawId() << " Pos = " << rechit.localPosition() << endl;
	    float mindist = 999999;
	    float dist;
	    PSimHit closest;
	    matched.clear();
	    matched = associate.associateHit(rechit);
	    if(!matched.empty()){
	      cout << " RPHI Strip detector =  " << myid << " Rechit = " << rechit.localPosition() << endl; 
	      if(matched.size()>1) cout << " matched = " << matched.size() << endl;
	      for(vector<PSimHit>::const_iterator m=matched.begin(); m<matched.end(); m++){
		cout << " simtrack ID = " << (*m).trackId() << " Simhit x = " << (*m).localPosition() << endl;
		dist = fabs(rechit.localPosition().x() - (*m).localPosition().x());
		if(dist<mindist){
		  mindist = dist;
		  closest = (*m);
		}
	      }  
	      cout << " Closest Simhit = " << closest.localPosition() << endl;
	    }
	    i++;
	  } 
	  
	  for(iterster=rechitsterRangeIteratorBegin;iterster!=rechitsterRangeIteratorEnd;++iterster){//loop on the rechit
	    SiStripRecHit2D const rechit=*iterster;
	    int i=0;
	    stripcounter++;
	    cout << stripcounter <<") Strip RecHit DetId " << detid.rawId() << " Pos = " << rechit.localPosition() << endl;
	    float mindist = 999999;
	    float dist;
	    PSimHit closest;
	    matched.clear();
	    matched = associate.associateHit(rechit);
	    if(!matched.empty()){
	      cout << " SAS Strip detector =  " << myid << " Rechit = " << rechit.localPosition() << endl; 
	      if(matched.size()>1) cout << " matched = " << matched.size() << endl;
	      for(vector<PSimHit>::const_iterator m=matched.begin(); m<matched.end(); m++){
		cout << " simtrack ID = " << (*m).trackId() << " Simhit x = " << (*m).localPosition() << endl;
		dist = fabs(rechit.localPosition().x() - (*m).localPosition().x());
		if(dist<mindist){
		  mindist = dist;
		  closest = (*m);
		}
	      }  
	      cout << " Closest Simhit = " << closest.localPosition() << endl;
	    }
	    i++;
	  } 
	  
	  for(itermatch=rechitmatchRangeIteratorBegin;itermatch!=rechitmatchRangeIteratorEnd;++itermatch){//loop on the rechit
	    SiStripMatchedRecHit2D const rechit=*itermatch;
	    int i=0;
	    stripcounter++;
	    cout << stripcounter <<") Strip RecHit DetId " << detid.rawId() << " Pos = " << rechit.localPosition() << endl;
	    float mindist = 999999;
	    float distx = 9999999;
	    float disty = 9999999;
	    float dist  = 9999999;
	    PSimHit closest;
	    matched.clear();
	    matched = associate.associateHit(rechit);
	    if(!matched.empty()){
	      cout << " MTC Strip detector =  " << myid << " Rechit = " << rechit.localPosition() << endl; 
	      if(matched.size()>1) cout << " matched = " << matched.size() << endl;
	      for(vector<PSimHit>::const_iterator m=matched.begin(); m<matched.end(); m++){
		cout << " simtrack ID = " << (*m).trackId() << " Simhit x = " << (*m).localPosition() << endl;
		
		distx = fabs(rechit.localPosition().x() - (*m).localPosition().x());
		disty = fabs(rechit.localPosition().y() - (*m).localPosition().y());
		dist = sqrt(distx*distx+disty*disty);
		if(dist<mindist){
		  mindist = dist;
		  closest = (*m);
		}
	      }  
	      cout << " Closest Simhit = " << closest.localPosition() << endl;
	    }
	    i++;
	  } 
	}
      }
    } 
    cout << " === calling end job " << endl;  
  }


TestAssociator::TestAssociator(edm::ParameterSet const& conf) : 
  conf_(conf),
  doPixel_( conf.getParameter<bool>("associatePixel") ),
  doStrip_( conf.getParameter<bool>("associateStrip") ) {
  cout << " Constructor " << endl;
 
}

  TestAssociator::~TestAssociator() 
  {
    cout << " Destructor " << endl;
  }


