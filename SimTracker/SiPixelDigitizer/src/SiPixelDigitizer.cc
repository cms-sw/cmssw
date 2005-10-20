// -*- C++ -*-
//
// Package:    SiPixelDigitizer
// Class:      SiPixelDigitizer
// 
/**\class SiPixelDigitizer SiPixelDigitizer.cc SimTracker/SiPixelDigitizer/src/SiPixelDigitizer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Michele Pioppi-INFN perugia
//         Created:  Mon Sep 26 11:08:32 CEST 2005
// $Id$
//
//


// system include files
#include <memory>
// user include files
#include "SimTracker/SiPixelDigitizer/interface/SiPixelDigitizer.h"
#include "SimTracker/SiPixelDigitizer/interface/SiPixelDigitizerAlgorithm.h"
//#include "SimTracker/SiPixelDigitizer/interface/PseudoHit.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigiCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "Geometry/Vector/interface/LocalPoint.h"
#include "Geometry/Vector/interface/LocalVector.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/TrackerSimAlgo/interface/PixelGeomDetType.h"

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
  SiPixelDigitizer::SiPixelDigitizer(const edm::ParameterSet& iConfig):
    _pixeldigialgo(iConfig) ,
    conf_(iConfig)
  {
    
    produces<PixelDigiCollection>();
   

  }

  
  SiPixelDigitizer::~SiPixelDigitizer()
  {}


  //
  // member functions
  //
  
  // ------------ method called to produce the data  ------------
  void
  SiPixelDigitizer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
  {
 
 
    // Step B: create empty output collection
    std::auto_ptr<PixelDigiCollection> output(new PixelDigiCollection);       

    // Step A: Create Inputs
 
    edm::ESHandle<TrackerGeom> pDD;
 
    iSetup.get<TrackerDigiGeometryRecord> ().get (pDD);
 
    int i=0;
    for(TrackerGeom::DetContainer::iterator iu = pDD->dets().begin(); iu != pDD->dets().end(); iu ++){

      //   const GeomDetUnit& iu(**iu);
      if (dynamic_cast<PixelGeomDetUnit*>((*iu))!=0){
	//	cout<<"bau"<<endl; 
	i++;
	if (i<10){
	  int idummy=0;

	  for (int i=0;i<3;i++) {
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
	  _pixeldigialgo.run(pseudoHitSingleContainer,*output,dynamic_cast<PixelGeomDetUnit*>((*iu)));
	}

      }
    }

    
 
    // Step D: write output to file
    iEvent.put(output);
  
  }
}
//define this as a plug-in

