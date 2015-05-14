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
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
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

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
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
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace CLHEP {
  class HepRandomEngine;
}


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
  SiPixelDigitizer::SiPixelDigitizer(const edm::ParameterSet& iConfig, edm::one::EDProducerBase& mixMod, edm::ConsumesCollector& iC):
    first(true),
    _pixeldigialgo(),
    hitsProducer(iConfig.getParameter<std::string>("hitsProducer")),
    trackerContainers(iConfig.getParameter<std::vector<std::string> >("RoutList")),
    geometryType(iConfig.getParameter<std::string>("PixGeometryType")),
    pilotBlades(iConfig.exists("enablePilotBlades")?iConfig.getParameter<bool>("enablePilotBlades"):false),
    NumberOfEndcapDisks(iConfig.exists("NumPixelEndcap")?iConfig.getParameter<int>("NumPixelEndcap"):2)
  {
    edm::LogInfo ("PixelDigitizer ") <<"Enter the Pixel Digitizer";
    
    const std::string alias ("simSiPixelDigis"); 
    
    mixMod.produces<edm::DetSetVector<PixelDigi> >().setBranchAlias(alias);
    mixMod.produces<edm::DetSetVector<PixelDigiSimLink> >().setBranchAlias(alias + "siPixelDigiSimLink");
    for(auto const& trackerContainer : trackerContainers) {
      edm::InputTag tag(hitsProducer, trackerContainer);
      iC.consumes<std::vector<PSimHit> >(edm::InputTag(hitsProducer, trackerContainer));
    }
    edm::Service<edm::RandomNumberGenerator> rng;
    if ( ! rng.isAvailable()) {
      throw cms::Exception("Configuration")
        << "SiPixelDigitizer requires the RandomNumberGeneratorService\n"
        "which is not present in the configuration file.  You must add the service\n"
        "in the configuration file or remove the modules that require it.";
    }

    _pixeldigialgo.reset(new SiPixelDigitizerAlgorithm(iConfig));
  }
  
  SiPixelDigitizer::~SiPixelDigitizer(){  
    edm::LogInfo ("PixelDigitizer ") <<"Destruct the Pixel Digitizer";
  }


  //
  // member functions
  //

  void
  SiPixelDigitizer::accumulatePixelHits(edm::Handle<std::vector<PSimHit> > hSimHits,
					size_t globalSimHitIndex,
					const unsigned int tofBin,
					CLHEP::HepRandomEngine* engine,
					edm::EventSetup const& iSetup) {
    if(hSimHits.isValid()) {
       std::set<unsigned int> detIds;
       std::vector<PSimHit> const& simHits = *hSimHits.product();
       edm::ESHandle<TrackerTopology> tTopoHand;
       iSetup.get<IdealGeometryRecord>().get(tTopoHand);
       const TrackerTopology *tTopo=tTopoHand.product();
       for(std::vector<PSimHit>::const_iterator it = simHits.begin(), itEnd = simHits.end(); it != itEnd; ++it, ++globalSimHitIndex) {
         unsigned int detId = (*it).detUnitId();
         if(detIds.insert(detId).second) {
           // The insert succeeded, so this detector element has not yet been processed.
	   assert(detectorUnits[detId]);
	   if(detectorUnits[detId] && detectorUnits[detId]->type().isTrackerPixel()) { // this test could be avoided and changed into a check of pixdet!=0
	     std::map<unsigned int, PixelGeomDetUnit const *>::iterator itDet = detectorUnits.find(detId);	     
	     if (itDet == detectorUnits.end()) continue;
             auto pixdet = itDet->second;
	     assert(pixdet !=0);
             //access to magnetic field in global coordinates
             GlobalVector bfield = pSetup->inTesla(pixdet->surface().position());
             LogDebug ("PixelDigitizer ") << "B-field(T) at " << pixdet->surface().position() << "(cm): " 
                                          << pSetup->inTesla(pixdet->surface().position());
             _pixeldigialgo->accumulateSimHits(it, itEnd, globalSimHitIndex, tofBin, pixdet, bfield, tTopo, engine);
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
    // Make sure that the first crossing processed starts indexing the sim hits from zero.
    // This variable is used so that the sim hits from all crossing frames have sequential
    // indices used to create the digi-sim link (if configured to do so) rather than starting
    // from zero for each crossing.
    crossingSimHitIndexOffset_.clear();

    _pixeldigialgo->initializeEvent();
    iSetup.get<TrackerDigiGeometryRecord>().get(geometryType, pDD);
    iSetup.get<IdealMagneticFieldRecord>().get(pSetup);
    edm::ESHandle<TrackerTopology> tTopoHand;
    iSetup.get<IdealGeometryRecord>().get(tTopoHand);
    const TrackerTopology *tTopo=tTopoHand.product();

    // FIX THIS! We only need to clear and (re)fill this map when the geometry type IOV changes.  Use ESWatcher to determine this.
    if(true) { // Replace with ESWatcher 
      detectorUnits.clear();
      for(TrackingGeometry::DetUnitContainer::const_iterator iu = pDD->detUnits().begin(); iu != pDD->detUnits().end(); ++iu) {
        unsigned int detId = (*iu)->geographicalId().rawId();
	if((*iu)->type().isTrackerPixel()) {
          auto pixdet = dynamic_cast<const PixelGeomDetUnit*>((*iu));
          assert(pixdet != 0);
	  if ((*iu)->subDetector()==GeomDetEnumerators::SubDetector::PixelEndcap) { // true ONLY for the phase 0 pixel deetctor
	    unsigned int disk = tTopo->layer(detId); // using the generic layer method
	    //if using pilot blades, then allowing it for current detector only
	    if ((disk == 3)&&((!pilotBlades)&&(NumberOfEndcapDisks == 2))) continue;
	  }
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
      unsigned int tofBin = PixelDigiSimLink::LowTof;
      if ((*i).find(std::string("HighTof")) != std::string::npos) tofBin = PixelDigiSimLink::HighTof;
      accumulatePixelHits(simHits, crossingSimHitIndexOffset_[tag.encode()], tofBin, randomEngine(iEvent.streamID()), iSetup);
      // Now that the hits have been processed, I'll add the amount of hits in this crossing on to
      // the global counter. Next time accumulateStripHits() is called it will count the sim hits
      // as though they were on the end of this collection.
      // Note that this is only used for creating digi-sim links (if configured to do so).
//       std::cout << "index offset, current hit count = " << crossingSimHitIndexOffset_[tag.encode()] << ", " << simHits->size() << std::endl;
      if( simHits.isValid() ) crossingSimHitIndexOffset_[tag.encode()]+=simHits->size();
    }
  }

  void
  SiPixelDigitizer::accumulate(PileUpEventPrincipal const& iEvent, edm::EventSetup const& iSetup, edm::StreamID const& streamID) {
    // Step A: Get Inputs
    for(vstring::const_iterator i = trackerContainers.begin(), iEnd = trackerContainers.end(); i != iEnd; ++i) {
      edm::Handle<std::vector<PSimHit> > simHits;
      edm::InputTag tag(hitsProducer, *i);

      iEvent.getByLabel(tag, simHits);
      unsigned int tofBin = PixelDigiSimLink::LowTof;
      if ((*i).find(std::string("HighTof")) != std::string::npos) tofBin = PixelDigiSimLink::HighTof;
      accumulatePixelHits(simHits, crossingSimHitIndexOffset_[tag.encode()], tofBin, randomEngine(streamID), iSetup);
      // Now that the hits have been processed, I'll add the amount of hits in this crossing on to
      // the global counter. Next time accumulateStripHits() is called it will count the sim hits
      // as though they were on the end of this collection.
      // Note that this is only used for creating digi-sim links (if configured to do so).
//       std::cout << "index offset, current hit count = " << crossingSimHitIndexOffset_[tag.encode()] << ", " << simHits->size() << std::endl;
      if( simHits.isValid() ) crossingSimHitIndexOffset_[tag.encode()]+=simHits->size();
    }
  }

  // ------------ method called to produce the data  ------------
  void
  SiPixelDigitizer::finalizeEvent(edm::Event& iEvent, const edm::EventSetup& iSetup) {

    edm::Service<edm::RandomNumberGenerator> rng;
    CLHEP::HepRandomEngine* engine = &rng->getEngine(iEvent.streamID());

    edm::ESHandle<TrackerTopology> tTopoHand;
    iSetup.get<IdealGeometryRecord>().get(tTopoHand);
    const TrackerTopology *tTopo=tTopoHand.product();

    std::vector<edm::DetSet<PixelDigi> > theDigiVector;
    std::vector<edm::DetSet<PixelDigiSimLink> > theDigiLinkVector;
 
    PileupInfo_ = getEventPileupInfo();
    _pixeldigialgo->calculateInstlumiFactor(PileupInfo_);   

    for(TrackingGeometry::DetUnitContainer::const_iterator iu = pDD->detUnits().begin(); iu != pDD->detUnits().end(); iu ++){
      
      if((*iu)->type().isTrackerPixel()) {

	//

        edm::DetSet<PixelDigi> collector((*iu)->geographicalId().rawId());
        edm::DetSet<PixelDigiSimLink> linkcollector((*iu)->geographicalId().rawId());
        
        
        _pixeldigialgo->digitize(dynamic_cast<const PixelGeomDetUnit*>((*iu)),
                                 collector.data,
                                 linkcollector.data,
				 tTopo,
                                 engine);
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

  CLHEP::HepRandomEngine* SiPixelDigitizer::randomEngine(edm::StreamID const& streamID) {
    unsigned int index = streamID.value();
    if(index >= randomEngines_.size()) {
      randomEngines_.resize(index + 1, nullptr);
    }
    CLHEP::HepRandomEngine* ptr = randomEngines_[index];
    if(!ptr) {
      edm::Service<edm::RandomNumberGenerator> rng;
      ptr = &rng->getEngine(streamID);
      randomEngines_[index] = ptr;
    }
    return ptr;
  }

}// end namespace cms::

