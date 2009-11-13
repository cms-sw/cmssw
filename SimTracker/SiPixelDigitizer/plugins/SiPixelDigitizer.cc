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
//   Modifications: Freya Blekman - Cornell University
//         Created:  Mon Sep 26 11:08:32 CEST 2005
// $Id: SiPixelDigitizer.cc,v 1.5 2009/10/16 09:27:37 fambrogl Exp $
//
//


// system include files
#include <memory>
// user include files
#include "SimTracker/SiPixelDigitizer/interface/SiPixelDigitizer.h"
#include "SimTracker/SiPixelDigitizer/interface/SiPixelDigitizerAlgorithm.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"

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
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

//Random Number
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CLHEP/Random/RandomEngine.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
//using namespace std;


namespace cms
{
  SiPixelDigitizer::SiPixelDigitizer(const edm::ParameterSet& iConfig):
    conf_(iConfig),first(true)
  {
    edm::LogInfo ("PixelDigitizer ") <<"Enter the Pixel Digitizer";
    
    std::string alias ( iConfig.getParameter<std::string>("@module_label") ); 
    
    produces<edm::DetSetVector<PixelDigi> >().setBranchAlias( alias );
    produces<edm::DetSetVector<PixelDigiSimLink> >().setBranchAlias ( alias + "siPixelDigiSimLink");
    trackerContainers.clear();
    trackerContainers = iConfig.getParameter<std::vector<std::string> >("ROUList");
    geometryType = iConfig.getParameter<std::string>("GeometryType");
    edm::Service<edm::RandomNumberGenerator> rng;
    if ( ! rng.isAvailable()) {
      throw cms::Exception("Configuration")
        << "SiPixelDigitizer requires the RandomNumberGeneratorService\n"
        "which is not present in the configuration file.  You must add the service\n"
        "in the configuration file or remove the modules that require it.";
    }
  
    rndEngine       = &(rng->getEngine());
    _pixeldigialgo = new SiPixelDigitizerAlgorithm(iConfig,(*rndEngine));

  }
  
  SiPixelDigitizer::~SiPixelDigitizer(){  
    edm::LogInfo ("PixelDigitizer ") <<"Destruct the Pixel Digitizer";
    delete _pixeldigialgo;
  }


  //
  // member functions
  //
  
  // ------------ method called to produce the data  ------------
  void
  SiPixelDigitizer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
  {

    if(first){
      _pixeldigialgo->init(iSetup);
      first = false;
    }

    // Step A: Get Inputs
    edm::Handle<CrossingFrame<PSimHit> > cf_simhit;
    std::vector<const CrossingFrame<PSimHit> *> cf_simhitvec;
    for(uint32_t i = 0; i< trackerContainers.size();i++){
      iEvent.getByLabel("mix",trackerContainers[i],cf_simhit);
      cf_simhitvec.push_back(cf_simhit.product());
    }
    
    std::auto_ptr<MixCollection<PSimHit> > allPixelTrackerHits(new MixCollection<PSimHit>(cf_simhitvec));

    edm::ESHandle<TrackerGeometry> pDD;
    
    iSetup.get<TrackerDigiGeometryRecord> ().get(geometryType,pDD);
 
    edm::ESHandle<MagneticField> pSetup;
    iSetup.get<IdealMagneticFieldRecord>().get(pSetup);

    //Loop on PSimHit
    SimHitMap.clear();

    MixCollection<PSimHit>::iterator isim;
    for (isim=allPixelTrackerHits->begin(); isim!= allPixelTrackerHits->end();isim++) {
      DetId detid=DetId((*isim).detUnitId());
      unsigned int subid=detid.subdetId();
      if ((subid==  PixelSubdetector::PixelBarrel) || (subid== PixelSubdetector::PixelEndcap)) {
	SimHitMap[(*isim).detUnitId()].push_back((*isim));
      }
    }

    // Step B: LOOP on PixelGeomDetUnit //
    for(TrackingGeometry::DetUnitContainer::const_iterator iu = pDD->detUnits().begin(); iu != pDD->detUnits().end(); iu ++){
      DetId idet=DetId((*iu)->geographicalId().rawId());
      unsigned int isub=idet.subdetId();
      
      
      if  ((isub==  PixelSubdetector::PixelBarrel) || (isub== PixelSubdetector::PixelEndcap)) {  
        
        
        //access to magnetic field in global coordinates
        GlobalVector bfield=pSetup->inTesla((*iu)->surface().position());
        LogDebug ("PixelDigitizer ") << "B-field(T) at "<<(*iu)->surface().position()<<"(cm): " 
                                     << pSetup->inTesla((*iu)->surface().position());
        //
        
        edm::DetSet<PixelDigi> collector((*iu)->geographicalId().rawId());
        edm::DetSet<PixelDigiSimLink> linkcollector((*iu)->geographicalId().rawId());
        
        
        collector.data=
          _pixeldigialgo->run(SimHitMap[(*iu)->geographicalId().rawId()],
                             dynamic_cast<PixelGeomDetUnit*>((*iu)),
                             bfield);
        if (collector.data.size()>0){
          theDigiVector.push_back(collector);
          
          //digisimlink
          if(SimHitMap[(*iu)->geographicalId().rawId()].size()>0){
              linkcollector.data=_pixeldigialgo->make_link();
              if (linkcollector.data.size()>0) theDigiLinkVector.push_back(linkcollector);
          }
          
        }
      }
      
    }
    
    // Step C: create collection with the cache vector of DetSet 
    std::auto_ptr<edm::DetSetVector<PixelDigi> > 
      output(new edm::DetSetVector<PixelDigi>(theDigiVector) );
    std::auto_ptr<edm::DetSetVector<PixelDigiSimLink> > 
      outputlink(new edm::DetSetVector<PixelDigiSimLink>(theDigiLinkVector) );

    // Step D: write output to file 
    iEvent.put(output);
    iEvent.put(outputlink);
  }




}// end namespace cms::

