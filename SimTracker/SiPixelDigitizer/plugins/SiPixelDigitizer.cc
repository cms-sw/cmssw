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
// $Id: SiPixelDigitizer.cc,v 1.7 2012/06/07 18:12:51 wmtan Exp $
//
//


// system include files
#include <memory>
#include <set>

// user include files
#include "SiPixelDigitizer.h"
#include "SiPixelDigitizerAlgorithm.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigiCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

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
  SiPixelDigitizer::SiPixelDigitizer(const edm::ParameterSet& iConfig, edm::EDProducer& mixMod):
    first(true),
    _pixeldigialgo(),
    hitsProducer(iConfig.getParameter<std::string>("hitsProducer")),
    trackerContainers(iConfig.getParameter<std::vector<std::string> >("ROUList")),
    geometryType(iConfig.getParameter<std::string>("GeometryType"))
  {
    edm::LogInfo ("PixelDigitizer ") <<"Enter the Pixel Digitizer";
    
    const std::string alias ("simSiPixelDigis"); 
    
    mixMod.produces<edm::DetSetVector<PixelDigi> >().setBranchAlias(alias);
    mixMod.produces<edm::DetSetVector<PixelDigiSimLink> >().setBranchAlias(alias + "siPixelDigiSimLink");
    edm::Service<edm::RandomNumberGenerator> rng;
    if ( ! rng.isAvailable()) {
      throw cms::Exception("Configuration")
        << "SiPixelDigitizer requires the RandomNumberGeneratorService\n"
        "which is not present in the configuration file.  You must add the service\n"
        "in the configuration file or remove the modules that require it.";
    }
  
    rndEngine       = &(rng->getEngine());
    _pixeldigialgo.reset(new SiPixelDigitizerAlgorithm(iConfig,(*rndEngine)));

  }
  
  SiPixelDigitizer::~SiPixelDigitizer(){  
    edm::LogInfo ("PixelDigitizer ") <<"Destruct the Pixel Digitizer";
  }


  //
  // member functions
  //

  void
  SiPixelDigitizer::accumulatePixelHits(edm::Handle<std::vector<PSimHit> > hSimHits) {
    if(hSimHits.isValid()) {
       std::set<unsigned int> detIds;
       std::vector<PSimHit> const& simHits = *hSimHits.product();
       for(std::vector<PSimHit>::const_iterator it = simHits.begin(), itEnd = simHits.end(); it != itEnd; ++it) {
         unsigned int detId = (*it).detUnitId();
         if(detIds.insert(detId).second) {
           // The insert succeeded, so this detector element has not yet been processed.
           unsigned int isub = DetId(detId).subdetId();
           if((isub == PixelSubdetector::PixelBarrel) || (isub == PixelSubdetector::PixelEndcap)) {
             PixelGeomDetUnit* pixdet = detectorUnits[detId];
             //access to magnetic field in global coordinates
             GlobalVector bfield = pSetup->inTesla(pixdet->surface().position());
             LogDebug ("PixelDigitizer ") << "B-field(T) at " << pixdet->surface().position() << "(cm): " 
                                          << pSetup->inTesla(pixdet->surface().position());
             _pixeldigialgo->accumulateSimHits(it, itEnd, pixdet, bfield);
           }
         }
       }
    }
  }
  
  void
  SiPixelDigitizer::initializeEvent(edm::Event const& e, edm::EventSetup const& iSetup) {
    if(first){
      _pixeldigialgo->init(iSetup);
      first = false;
    }
    _pixeldigialgo->initializeEvent();
    iSetup.get<TrackerDigiGeometryRecord>().get(geometryType, pDD);
    iSetup.get<IdealMagneticFieldRecord>().get(pSetup);

    // FIX THIS! We only need to clear and (re)fill this map when the geometry type IOV changes.  Use ESWatcher to determine this.
    if(true) { // Replace with ESWatcher 
      detectorUnits.clear();
      for(TrackingGeometry::DetUnitContainer::const_iterator iu = pDD->detUnits().begin(); iu != pDD->detUnits().end(); ++iu) {
        unsigned int detId = (*iu)->geographicalId().rawId();
        DetId idet=DetId(detId);
        unsigned int isub=idet.subdetId();
        if((isub == PixelSubdetector::PixelBarrel) || (isub == PixelSubdetector::PixelEndcap)) {  
          PixelGeomDetUnit* pixdet = dynamic_cast<PixelGeomDetUnit*>((*iu));
          assert(pixdet != 0);
          detectorUnits.insert(std::make_pair(detId, pixdet));
        }
      }
    }
  }

  void
  SiPixelDigitizer::accumulate(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
    // Step A: Get Inputs
    for(vstring::const_iterator i = trackerContainers.begin(), iEnd = trackerContainers.end(); i != iEnd; ++i) {
      edm::Handle<std::vector<PSimHit> > simHits;
      edm::InputTag tag(hitsProducer, *i);

      iEvent.getByLabel(tag, simHits);
      accumulatePixelHits(simHits);
    }
  }

  void
  SiPixelDigitizer::accumulate(PileUpEventPrincipal const& iEvent, edm::EventSetup const& iSetup) {
    // Step A: Get Inputs
    for(vstring::const_iterator i = trackerContainers.begin(), iEnd = trackerContainers.end(); i != iEnd; ++i) {
      edm::Handle<std::vector<PSimHit> > simHits;
      edm::InputTag tag(hitsProducer, *i);

      iEvent.getByLabel(tag, simHits);
      accumulatePixelHits(simHits);
    }
  }

  // ------------ method called to produce the data  ------------
  void
  SiPixelDigitizer::finalizeEvent(edm::Event& iEvent, const edm::EventSetup& iSetup) {

    std::vector<edm::DetSet<PixelDigi> > theDigiVector;
    std::vector<edm::DetSet<PixelDigiSimLink> > theDigiLinkVector;

    for(TrackingGeometry::DetUnitContainer::const_iterator iu = pDD->detUnits().begin(); iu != pDD->detUnits().end(); iu ++){
      DetId idet=DetId((*iu)->geographicalId().rawId());
      unsigned int isub=idet.subdetId();
      
      if((isub == PixelSubdetector::PixelBarrel) || (isub == PixelSubdetector::PixelEndcap)) {  
        
        //
        
        edm::DetSet<PixelDigi> collector((*iu)->geographicalId().rawId());
        edm::DetSet<PixelDigiSimLink> linkcollector((*iu)->geographicalId().rawId());
        
        
        _pixeldigialgo->digitize(dynamic_cast<PixelGeomDetUnit*>((*iu)),
                                 collector.data,
                                 linkcollector.data);
        if(collector.data.size() > 0) {
          theDigiVector.push_back(std::move(collector));
        }
        if(linkcollector.data.size() > 0) {
          theDigiLinkVector.push_back(std::move(linkcollector));
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

