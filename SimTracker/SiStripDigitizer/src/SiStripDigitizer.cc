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
// Original Author:  Andrea Giammanco
//         Created:  Mon Sep 26 11:08:32 CEST 2005
// $Id: SiStripDigitizer.cc,v 1.1 2005/10/20 14:09:11 giamman Exp $
//
//


// system include files
#include <memory>
// user include files
#include "SimTracker/SiStripDigitizer/interface/SiStripDigitizer.h"
#include "SimTracker/SiStripDigitizer/interface/SiStripDigitizerAlgorithm.h"
//#include "SimTracker/SiStripDigitizer/interface/PseudoHit.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/SiStripDigi/interface/StripDigi.h"
#include "DataFormats/SiStripDigi/interface/StripDigiCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "Geometry/Vector/interface/LocalPoint.h"
#include "Geometry/Vector/interface/LocalVector.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/TrackerSimAlgo/interface/StripGeomDetType.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
using namespace std;
namespace cms
{
  SiStripDigitizer::SiStripDigitizer(const edm::ParameterSet& iConfig):
    _stripdigialgo(iConfig) ,
    conf_(iConfig)
  {
    
    produces<StripDigiCollection>();
   

  }

  
  SiStripDigitizer::~SiStripDigitizer()
  {}


  //
  // member functions
  //
  
  // ------------ method called to produce the data  ------------
  void
  SiStripDigitizer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
  {
 
 
    // Step B: create empty output collection
    std::auto_ptr<StripDigiCollection> output(new StripDigiCollection);       

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
	  cout << "--- Accessing module number " << i << endl;
	  int idummy=0;

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
	  _stripdigialgo.run(pseudoHitSingleContainer,*output,dynamic_cast<StripGeomDetUnit*>((*iu)));
	}

      }
    }

    
 
    // Step D: write output to file
    iEvent.put(output);
    cout << "Output written to file!" << endl;
  
  }
}
//define this as a plug-in

