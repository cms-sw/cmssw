// -*- C++ -*-
//
// Package:    SiStripDigitizer
// Class:      SiStripDigitizer
// 
/**\class SiStripDigitizer SiStripDigitizer.cc SimTracker/SiStripDigitizer/src/SiStripDigitizer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea GIAMMANCO
//         Created:  Thu Sep 22 14:23:22 CEST 2005
// $Id: SiStripDigitizer.cc,v 1.21 2006/05/23 08:43:16 fambrogl Exp $
//
//


// system include files
#include <memory>

#include "SimTracker/SiStripDigitizer/interface/SiStripDigitizer.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include <cstdlib> // I need it for random numbers

//needed for the geometry:
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
//needed for the magnetic field:
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
using namespace std;

namespace cms
{

  SiStripDigitizer::SiStripDigitizer(const edm::ParameterSet& conf) : 
    //   stripDigitizer_(conf) ,
    conf_(conf)
  {
    produces<edm::DetSetVector<SiStripDigi> >();
    produces<edm::DetSetVector<StripDigiSimLink> >();
    trackerContainers.clear();
    trackerContainers = conf.getParameter<std::vector<std::string> >("ROUList");
  }

  // Virtual destructor needed.
  SiStripDigitizer::~SiStripDigitizer() { }  

  // Functions that gets called by framework every event
  void SiStripDigitizer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
  {
    // Step A: Get Inputs
    edm::Handle<CrossingFrame> cf;
    iEvent.getByType(cf);

    std::auto_ptr<MixCollection<PSimHit> > allTrackerHits(new MixCollection<PSimHit>(cf.product(),trackerContainers));

    //Loop on PSimHit
    SimHitMap.clear();
    
    MixCollection<PSimHit>::iterator isim;
    for (isim=allTrackerHits->begin(); isim!= allTrackerHits->end();isim++) {
      SimHitMap[(*isim).detUnitId()].push_back((*isim));
    }
    
    edm::ESHandle<TrackerGeometry> pDD;
 
    iSetup.get<TrackerDigiGeometryRecord> ().get (pDD);
 
    edm::ESHandle<MagneticField> pSetup;
    iSetup.get<IdealMagneticFieldRecord>().get(pSetup);

  
    // Step B: LOOP on StripGeomDetUnit //
    
    theAlgoMap.clear();
    theDigiVector.reserve(10000);
    theDigiVector.clear();

    theDigiLinkVector.reserve(10000);
    theDigiLinkVector.clear();

    for(TrackingGeometry::DetUnitContainer::const_iterator iu = pDD->detUnits().begin(); iu != pDD->detUnits().end(); iu ++){
      
      GlobalVector bfield=pSetup->inTesla((*iu)->surface().position());
      
      StripGeomDetUnit* sgd = dynamic_cast<StripGeomDetUnit*>((*iu));
      if (sgd != 0){

	edm::DetSet<SiStripDigi> collector((*iu)->geographicalId().rawId());
	edm::DetSet<StripDigiSimLink> linkcollector((*iu)->geographicalId().rawId());
	
	if(theAlgoMap.find(&(sgd->type())) == theAlgoMap.end()) {
	  theAlgoMap[&(sgd->type())] = boost::shared_ptr<SiStripDigitizerAlgorithm>(new SiStripDigitizerAlgorithm(conf_, sgd));
	}
	
	collector.data= ((theAlgoMap.find(&(sgd->type())))->second)->run(SimHitMap[(*iu)->geographicalId().rawId()], sgd, bfield);
	
	if (collector.data.size()>0){
	  
	  theDigiVector.push_back(collector);
	  
	  //digisimlink
	  if(SimHitMap[(*iu)->geographicalId().rawId()].size()>0){
	    linkcollector.data = ((theAlgoMap.find(&(sgd->type())))->second)->make_link();
	    if (linkcollector.data.size()>0)   theDigiLinkVector.push_back(linkcollector);
	  }
	}
      }
    }
    
    // Step C: create empty output collection
    std::auto_ptr<edm::DetSetVector<SiStripDigi> > output(new edm::DetSetVector<SiStripDigi>(theDigiVector) );
    std::auto_ptr<edm::DetSetVector<StripDigiSimLink> > outputlink(new edm::DetSetVector<StripDigiSimLink>(theDigiLinkVector) );

    
    
    // Step D: write output to file
    iEvent.put(output);
    iEvent.put(outputlink);
    
  
  }





}
//define this as a plug-in

