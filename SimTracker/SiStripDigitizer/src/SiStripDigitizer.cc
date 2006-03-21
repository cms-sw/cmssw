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
// $Id: SiStripDigitizer.cc,v 1.12 2006/03/13 15:25:36 azzi Exp $
//
//


// system include files
#include <memory>


#include "SimTracker/SiStripDigitizer/interface/SiStripDigitizer.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/SiStripDigi/interface/StripDigiCollection.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLinkCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include <cstdlib> // I need it for random numbers

//needed for the geometry:
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerBaseAlgo/interface/GeometricDet.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/TrackerSimAlgo/interface/PixelGeomDetType.h"
#include "Geometry/TrackerSimAlgo/interface/StripGeomDetType.h"
#include "Geometry/TrackerSimAlgo/interface/StripGeomDetUnit.h"
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

    produces<StripDigiCollection>();
    produces<StripDigiSimLinkCollection>();
  }

  // Virtual destructor needed.
  SiStripDigitizer::~SiStripDigitizer() { }  

  // Functions that gets called by framework every event
  void SiStripDigitizer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
  {
    // Step A: Get Inputs
    theStripHits.clear();
    edm::Handle<edm::PSimHitContainer> TIBHitsLowTof;
    edm::Handle<edm::PSimHitContainer> TIBHitsHighTof;
    edm::Handle<edm::PSimHitContainer> TIDHitsLowTof;
    edm::Handle<edm::PSimHitContainer> TIDHitsHighTof;
    edm::Handle<edm::PSimHitContainer> TOBHitsLowTof;
    edm::Handle<edm::PSimHitContainer> TOBHitsHighTof;
    edm::Handle<edm::PSimHitContainer> TECHitsLowTof;
    edm::Handle<edm::PSimHitContainer> TECHitsHighTof;

    iEvent.getByLabel("SimG4Object","TrackerHitsTIBLowTof", TIBHitsLowTof);
    iEvent.getByLabel("SimG4Object","TrackerHitsTIBHighTof", TIBHitsHighTof);
    iEvent.getByLabel("SimG4Object","TrackerHitsTIDLowTof", TIDHitsLowTof);
    iEvent.getByLabel("SimG4Object","TrackerHitsTIDHighTof", TIDHitsHighTof);
    iEvent.getByLabel("SimG4Object","TrackerHitsTOBLowTof", TOBHitsLowTof);
    iEvent.getByLabel("SimG4Object","TrackerHitsTOBHighTof", TOBHitsHighTof);
    iEvent.getByLabel("SimG4Object","TrackerHitsTECLowTof", TECHitsLowTof);
    iEvent.getByLabel("SimG4Object","TrackerHitsTECHighTof", TECHitsHighTof);
    
    theStripHits.insert(theStripHits.end(), TIBHitsLowTof->begin(), TIBHitsLowTof->end()); 
    theStripHits.insert(theStripHits.end(), TIBHitsHighTof->begin(), TIBHitsHighTof->end());
    theStripHits.insert(theStripHits.end(), TIDHitsLowTof->begin(), TIDHitsLowTof->end()); 
    theStripHits.insert(theStripHits.end(), TIDHitsHighTof->begin(), TIDHitsHighTof->end());
    theStripHits.insert(theStripHits.end(), TOBHitsLowTof->begin(), TOBHitsLowTof->end()); 
    theStripHits.insert(theStripHits.end(), TOBHitsHighTof->begin(), TOBHitsHighTof->end());
    theStripHits.insert(theStripHits.end(), TECHitsLowTof->begin(), TECHitsLowTof->end()); 
    theStripHits.insert(theStripHits.end(), TECHitsHighTof->begin(), TECHitsHighTof->end());

    // Step B: create empty output collection
    std::auto_ptr<StripDigiCollection> output(new StripDigiCollection);
    std::auto_ptr<StripDigiSimLinkCollection> outputlink(new StripDigiSimLinkCollection);

    //Loop on PSimHit
    SimHitMap.clear();
    
    for (std::vector<PSimHit>::iterator isim = theStripHits.begin();
	 isim != theStripHits.end(); ++isim){
      /*
	DetId detid=DetId((*isim).detUnitId());
	unsigned int subid=detid.subdetId();
	if ((subid==  StripSubdetector::TIB) || 
	(subid== StripSubdetector::TOB)  ||
	(subid==  StripSubdetector::TEC) || 
	(subid== StripSubdetector::TID)) {
      */
      SimHitMap[(*isim).detUnitId()].push_back((*isim));
      //      }
    }
    
    
    // Temporary: generate random collections of pseudo-hits:
    //PseudoHitContainer pseudoHitContainer;// for some reason this class isn't recognized!!!
    
    
    edm::ESHandle<TrackerGeometry> pDD;
 
    iSetup.get<TrackerDigiGeometryRecord> ().get (pDD);
 
    edm::ESHandle<MagneticField> pSetup;
    iSetup.get<IdealMagneticFieldRecord>().get(pSetup);

  
    // Step C: LOOP on StripGeomDetUnit //
    
    theAlgoMap.clear();
    for(TrackingGeometry::DetUnitContainer::const_iterator iu = pDD->detUnits().begin(); iu != pDD->detUnits().end(); iu ++){
      
      GlobalVector bfield=pSetup->inTesla((*iu)->surface().position());
      
      StripGeomDetUnit* sgd = dynamic_cast<StripGeomDetUnit*>((*iu));
      if (sgd != 0){
	
	collector.clear();
	linkcollector.clear();
	
	if(theAlgoMap.find(&(sgd->type())) == theAlgoMap.end()) {
	  theAlgoMap[&(sgd->type())] = new SiStripDigitizerAlgorithm(conf_, sgd);
	}
	
	collector= ((theAlgoMap.find(&(sgd->type())))->second)->run(SimHitMap[(*iu)->geographicalId().rawId()], sgd, bfield);
	
	if (collector.size()>0){
	  StripDigiCollection::Range outputRange;
	  
	  outputRange.first = collector.begin();
	  outputRange.second = collector.end();
	  output->put(outputRange,(*iu)->geographicalId().rawId());
	  
	  //digisimlink
	  if(SimHitMap[(*iu)->geographicalId().rawId()].size()>0){
	    StripDigiSimLinkCollection::Range outputlinkRange;
	    linkcollector= ((theAlgoMap.find(&(sgd->type())))->second)->make_link();
	    outputlinkRange.first = linkcollector.begin();	  
	    outputlinkRange.second = linkcollector.end();
	    outputlink->put(outputlinkRange,(*iu)->geographicalId().rawId());
	  }
	}	
	
      }
      
    }
    
    
    
    
    // Step D: write output to file
    iEvent.put(output);
    iEvent.put(outputlink);
    
  
  }





}
//define this as a plug-in

