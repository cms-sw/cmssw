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
// $Id: SiPixelDigitizer.cc,v 1.10 2006/01/29 14:44:55 pioppi Exp $
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

#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLinkCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "Geometry/Vector/interface/LocalPoint.h"
#include "Geometry/Vector/interface/LocalVector.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/TrackerSimAlgo/interface/PixelGeomDetType.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
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
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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
    conf_(iConfig),
    _pixeldigialgo(iConfig) 
  {
    edm::LogInfo ("PixelDigitizer ") <<"Enter the Pixel Digitizer";
    produces<PixelDigiCollection>();
    produces<PixelDigiSimLinkCollection>();

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


    edm::Handle<edm::PSimHitContainer> PixelBarrelHitsLowTof;
    edm::Handle<edm::PSimHitContainer> PixelBarrelHitsHighTof;
    edm::Handle<edm::PSimHitContainer> PixelEndcapHitsLowTof;
    edm::Handle<edm::PSimHitContainer> PixelEndcapHitsHighTof;

    iEvent.getByLabel("r","TrackerHitsPixelBarrelLowTof", PixelBarrelHitsLowTof);
    iEvent.getByLabel("r","TrackerHitsPixelBarrelHighTof", PixelBarrelHitsHighTof);
    iEvent.getByLabel("r","TrackerHitsPixelEndcapLowTof", PixelEndcapHitsLowTof);
    iEvent.getByLabel("r","TrackerHitsPixelEndcapHighTof", PixelEndcapHitsHighTof);

    thePixelHits.insert(thePixelHits.end(), PixelBarrelHitsLowTof->begin(), PixelBarrelHitsLowTof->end()); 
    thePixelHits.insert(thePixelHits.end(), PixelBarrelHitsHighTof->begin(), PixelBarrelHitsHighTof->end());
    thePixelHits.insert(thePixelHits.end(), PixelEndcapHitsLowTof->begin(), PixelEndcapHitsLowTof->end()); 
    thePixelHits.insert(thePixelHits.end(), PixelEndcapHitsHighTof->begin(), PixelEndcapHitsHighTof->end());
    // Step B: create empty output collection
    std::auto_ptr<PixelDigiCollection> output(new PixelDigiCollection);       

    std::auto_ptr<PixelDigiSimLinkCollection> outputlink(new PixelDigiSimLinkCollection);
 
    edm::ESHandle<TrackingGeometry> pDD;
    
    iSetup.get<TrackerDigiGeometryRecord> ().get(pDD);
 
    edm::ESHandle<MagneticField> pSetup;
    iSetup.get<IdealMagneticFieldRecord>().get(pSetup);

    //Loop on PSimHit
    SimHitMap.clear();

    for (std::vector<PSimHit>::iterator isim = thePixelHits.begin();
	 isim != thePixelHits.end(); ++isim){
      DetId detid=DetId((*isim).detUnitId());
      unsigned int subid=detid.subdetId();
      if ((subid==  PixelSubdetector::PixelBarrel) || (subid== PixelSubdetector::PixelEndcap)) {
	SimHitMap[(*isim).detUnitId()].push_back((*isim));
      }
    }

    // Step C: LOOP on PixelGeomDetUnit //
    for(TrackingGeometry::DetContainer::const_iterator iu = pDD->dets().begin(); iu != pDD->dets().end(); iu ++){
  
      GlobalVector bfield=pSetup->inTesla((*iu)->surface().position());

      if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 1 ) {
	LogDebug ("PixelDigitizer ") << "B-field(T) at "<<(*iu)->surface().position()<<"(cm): " 
		  << pSetup->inTesla((*iu)->surface().position());
      }
 
      if (dynamic_cast<PixelGeomDetUnit*>((*iu))!=0){

	collector.clear();
	linkcollector.clear();

	collector= _pixeldigialgo.run(SimHitMap[(*iu)->geographicalId().rawId()],
				      dynamic_cast<PixelGeomDetUnit*>((*iu)),
				      bfield);
	if (collector.size()>0){

	  PixelDigiCollection::Range outputRange;	  
	  outputRange.first = collector.begin();
	  outputRange.second = collector.end();
	  output->put(outputRange,(*iu)->geographicalId().rawId());
	  
	  //digisimlink
	  if(SimHitMap[(*iu)->geographicalId().rawId()].size()>0){
	    PixelDigiSimLinkCollection::Range outputlinkRange;
	    linkcollector=_pixeldigialgo.make_link();
	    outputlinkRange.first = linkcollector.begin();
	    outputlinkRange.second = linkcollector.end();
	    outputlink->put(outputlinkRange,(*iu)->geographicalId().rawId());
	  }

	}
      }

    }

    // Step D: write output to file
    iEvent.put(outputlink);
    iEvent.put(output);
  }
}


