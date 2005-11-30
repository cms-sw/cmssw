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
// $Id: SiStripDigitizer.cc,v 1.4 2005/11/12 17:33:49 giamman Exp $
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
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include <cstdlib> // I need it for random numbers

//needed for the geometry:
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
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

using namespace std;

namespace cms
{

  SiStripDigitizer::SiStripDigitizer(const edm::ParameterSet& conf) : 
    stripDigitizer_(conf) ,
    conf_(conf)
  {
    //    numStrips=conf_.getParameter<int>("NumStrips"); // temporary!

    produces<StripDigiCollection>();
  }

  // Virtual destructor needed.
  SiStripDigitizer::~SiStripDigitizer() { }  

  // Functions that gets called by framework every event
  void SiStripDigitizer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
  {
    //   using namespace edm;


   // Step B: create empty output collection
   std::auto_ptr<StripDigiCollection> output(new StripDigiCollection);



    // Step A: Get Inputs 
    /*
    edm::Handle<PSimHit> simHitsTrackerHitsTECHighTof;
    e.getByLabel("TrackerHitsTECHighTof",simHitsTrackerHitsTECHighTof);
    edm::Handle<PSimHit> simHitsTrackerHitsTECLowTof;
    e.getByLabel("TrackerHitsTECLowTof",simHitsTrackerHitsTECLowTof);

    edm::Handle<PSimHit> simHitsTrackerHitsTIBHighTof;
    e.getByLabel("TrackerHitsTIBHighTof",simHitsTrackerHitsTIBHighTof);
    edm::Handle<PSimHit> simHitsTrackerHitsTIBLowTof;
    e.getByLabel("TrackerHitsTIBLowTof",simHitsTrackerHitsTIBLowTof);

    edm::Handle<PSimHit> simHitsTrackerHitsTIDHighTof;
    e.getByLabel("TrackerHitsTIDHighTof",simHitsTrackerHitsTIDHighTof);
    edm::Handle<PSimHit> simHitsTrackerHitsTIDLowTof;
    e.getByLabel("TrackerHitsTIDLowTof",simHitsTrackerHitsTIDLowTof);

    edm::Handle<PSimHit> simHitsTrackerHitsTOBHighTof;
    e.getByLabel("TrackerHitsTOBHighTof",simHitsTrackerHitsTOBHighTof);
    edm::Handle<PSimHit> simHitsTrackerHitsTOBLowTof;
    e.getByLabel("TrackerHitsTOBLowTof",simHitsTrackerHitsTOBLowTof);
    */
    // Temporary: generate random collections of pseudo-hits:
    //PseudoHitContainer pseudoHitContainer;// for some reason this class isn't recognized!!!

    // Step A: Create Inputs
 
    edm::ESHandle<TrackingGeometry> pDD;
 
    iSetup.get<TrackerDigiGeometryRecord> ().get (pDD);
 
    edm::ESHandle<MagneticField> pSetup;
    iSetup.get<IdealMagneticFieldRecord>().get(pSetup);

    int i=0;
    for(TrackingGeometry::DetContainer::iterator iu = pDD->dets().begin(); iu != pDD->dets().end(); iu ++){

      GlobalVector bfield=pSetup->inTesla((*iu)->surface().position());

      //   const GeomDetUnit& iu(**iu);
      if (dynamic_cast<StripGeomDetUnit*>((*iu))!=0){
	i++;
	if (i<10){
	  int idummy=0;

	  // create 3 hits:
	  for (int j=0;j<3;j++) {
	    irandom1 = rand();
	    irandom2 = rand();
	    irandom3 = rand();
	    
	    
	    xexrand = (1.6*rand()/RAND_MAX)-0.8;
	    xentrand = (1.6*rand()/RAND_MAX)-0.8;	 
	    yexrand= (6.4*rand()/RAND_MAX)-3.2;
	    yentrand= (6.4*rand()/RAND_MAX)-3.2;
	    zexrand= (0.028*rand()/RAND_MAX)-0.014;
	    zentrand= (0.028*rand()/RAND_MAX)-0.014;
	    frandom3 = rand();
	    frandom4 = rand();
	    frandom5 = (20*rand()/RAND_MAX)+70;
	    angrandom1 = 3.14*rand()/RAND_MAX;
	    angrandom2 = 6.28*rand()/RAND_MAX;
	    Local3DPoint exit(xexrand,yexrand,zexrand);
	    Local3DPoint entry(xentrand,yentrand,zentrand);
	    //Local3DPoint exit();
	    int hh=1298553623;
	    
	    PSimHit *pseudoHit = new PSimHit(entry,exit ,frandom3, frandom4, frandom5, irandom1, hh, irandom3, angrandom1, angrandom2, idummy);

	    pseudoHitSingleContainer.push_back(pseudoHit);
	  }
	  stripDigitizer_.run(pseudoHitSingleContainer,*output,dynamic_cast<StripGeomDetUnit*>((*iu)),bfield);
	}

      }
    }

    


    std::cout << "pippo " << endl;
    // Step D: write output to file
    iEvent.put(output);
    std::cout << "pluto " << endl;

  
  }

}
//define this as a plug-in

