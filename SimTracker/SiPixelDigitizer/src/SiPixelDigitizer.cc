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
// $Id: SiPixelDigitizer.cc,v 1.3 2005/11/23 19:33:45 pioppi Exp $
//
//


// system include files
#include <memory>
// user include files
#include "SimTracker/SiPixelDigitizer/interface/SiPixelDigitizer.h"
#include "SimTracker/SiPixelDigitizer/interface/SiPixelDigitizerAlgorithm.h"

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
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/TrackerSimAlgo/interface/PixelGeomDetType.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"


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
    // Step A: Get Inputs
    thePixelHits.clear();

    edm::Handle<edm::PSimHitContainer> PixelHits;
    iEvent.getByLabel("r","TrackerHitsLowTof", PixelHits);
    //MP waiting for container bug fix

    thePixelHits.insert(thePixelHits.end(), PixelHits->begin(), PixelHits->end()); 

    // Step B: create empty output collection
    std::auto_ptr<PixelDigiCollection> output(new PixelDigiCollection);       

 
 
    edm::ESHandle<TrackingGeometry> pDD;
    
    iSetup.get<TrackerDigiGeometryRecord> ().get(pDD);
 
    edm::ESHandle<MagneticField> pSetup;
    iSetup.get<IdealMagneticFieldRecord>().get(pSetup);
    // Step C: LOOP on PixelGeomDetUnit //

    for(TrackingGeometry::DetContainer::iterator iu = pDD->dets().begin(); iu != pDD->dets().end(); iu ++){
  
      GlobalVector bfield=pSetup->inTesla((*iu)->surface().position());




      if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 1 ) {
	std::cout << "B-field(T) at "<<(*iu)->surface().position()<<"(cm): " 
		  << pSetup->inTesla((*iu)->surface().position()) << std::endl;
      }

      if (dynamic_cast<PixelGeomDetUnit*>((*iu))!=0){

	detPixelHits.clear();
	//Loop on SimHit collections
	for (std::vector<PSimHit>::iterator isim = thePixelHits.begin();
	     isim != thePixelHits.end(); ++isim){
		  if((*iu)->geographicalId().rawId()==(*isim).detUnitId()){
		    detPixelHits.push_back((*isim));
		  }
	}

	//take back the vector of PixelDigi 
	collector= _pixeldigialgo.run(detPixelHits,dynamic_cast<PixelGeomDetUnit*>((*iu)),bfield);	
	PixelDigiCollection::Range outputRange;
	outputRange.first = collector.begin();
	outputRange.second = collector.end();
       	output->put(outputRange,(*iu)->geographicalId().rawId());
	//
      }

    }
    // Step D: write output to file
    iEvent.put(output);
  }
}


