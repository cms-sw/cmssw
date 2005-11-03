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
// $Id$
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
//#include "SimTracker/SiStripDigitizer/interface/PseudoHitContainer.h"
//#include "SimTracker/SiStripDigitizer/interface/PseudoHit.h"
#include <cstdlib> // I need it for random numbers

//needed for the geometry:
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerSimAlgo/interface/TrackerGeom.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerBaseAlgo/interface/GeometricDet.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/TrackerSimAlgo/interface/PixelGeomDetType.h"
#include "Geometry/TrackerSimAlgo/interface/StripGeomDetType.h"
#include "Geometry/TrackerSimAlgo/interface/StripGeomDetUnit.h"
using namespace std;

namespace cms
{

  SiStripDigitizer::SiStripDigitizer(edm::ParameterSet const& conf) : 
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
   using namespace edm;


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
 
    edm::ESHandle<TrackerGeom> pDD;
 
    iSetup.get<TrackerDigiGeometryRecord> ().get (pDD);
 
    int i=0;
    for(TrackerGeom::DetContainer::iterator iu = pDD->dets().begin(); iu != pDD->dets().end(); iu ++){

      //   const GeomDetUnit& iu(**iu);
      if (dynamic_cast<StripGeomDetUnit*>((*iu))!=0){
	//	cout<<"bau"<<endl; 
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
	  stripDigitizer_.run(pseudoHitSingleContainer,*output,dynamic_cast<StripGeomDetUnit*>((*iu)));
	}

      }
    }

    



    /*
    std::vector<PseudoHit*> pseudoHitSingleContainer;
    int idummy=0;
    for (int i=0;i<numStrips;i++) {
      irandom1 = rand(); // particle type
      irandom2 = rand(); // detId
      irandom3 = rand(); // trackId
      frandom1 = rand(); // entry
      frandom2 = rand(); // exit
      frandom3 = rand(); // pabs
      Local3DPoint entryVector(frandom1,frandom2,0.);
      Local3DPoint exitVector(frandom1,frandom2,100.);
      frandom4 = 110.*rand()/RAND_MAX; // tof
      frandom5 = rand(); // eloss
      angrandom1 = 3.14*rand()/RAND_MAX; // theta
      angrandom2 = 6.28*rand()/RAND_MAX; // phi
      //      std::cout << "a " << irandom1 << " "  << irandom2 << " "  << irandom3 << " " << endl;
      //      std::cout << "b " << frandom1 << " "  << frandom2 << " "  << frandom3 << " "  << frandom4 << " "  << frandom5 << " " << endl;
      //      std::cout << "c " << angrandom1 << " "  << angrandom2 << " " << endl;
      //      std::cout << "i:  "  << i << endl;
      PseudoHit* pseudoHit = new PseudoHit(entryVector, exitVector ,frandom3, frandom4, frandom5, irandom1, irandom2, irandom3, angrandom1, angrandom2, idummy);
      pseudoHitSingleContainer.push_back(pseudoHit);
      delete pseudoHit;
    }

    // Step C: Invoke the StripDigi conversion algorithm
    //    stripDigitizer_.run(simHitsTrackerHitsTOBLowTof.product(),*output);
    stripDigitizer_.run(pseudoHitSingleContainer,*output);
    */

    std::cout << "pippo " << endl;
    // Step D: write output to file
    iEvent.put(output);
    std::cout << "pluto " << endl;

  
  }

}


//define this as a plug-in
//DEFINE_FWK_MODULE(SiStripDigitizer)
