// File: SiStripDigitizerAlgorithm.cc
// Description:  Class for digitization.

// Modified 15/May/2013 mark.grimes@bristol.ac.uk - Modified so that the digi-sim link has the correct
// index for the sim hits stored. It was previously always set to zero (I won't mention that it was
// me who originally wrote that).

// system include files
#include <memory>

#include "SimTracker/Common/interface/SimHitSelectorFromDB.h"

#include "SiStripDigitizer.h"
#include "SiStripDigitizerAlgorithm.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
//needed for the geometry:
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
//needed for the magnetic field:
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

//Data Base infromations
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibTracker/Records/interface/SiStripDependentRecords.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"

//Random Number
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"

SiStripDigitizer::SiStripDigitizer(const edm::ParameterSet& conf, edm::stream::EDProducerBase& mixMod, edm::ConsumesCollector& iC) : 
  gainLabel(conf.getParameter<std::string>("Gain")),
  hitsProducer(conf.getParameter<std::string>("hitsProducer")),
  trackerContainers(conf.getParameter<std::vector<std::string> >("ROUList")),
  ZSDigi(conf.getParameter<edm::ParameterSet>("DigiModeList").getParameter<std::string>("ZSDigi")),
  SCDigi(conf.getParameter<edm::ParameterSet>("DigiModeList").getParameter<std::string>("SCDigi")),
  VRDigi(conf.getParameter<edm::ParameterSet>("DigiModeList").getParameter<std::string>("VRDigi")),
  PRDigi(conf.getParameter<edm::ParameterSet>("DigiModeList").getParameter<std::string>("PRDigi")),
  geometryType(conf.getParameter<std::string>("GeometryType")),
  std::unique_ptr< edm::DetSetVector<StripDigiSimLink> > pOutputDigiSimLink( new edm::DetSetVector<StripDigiSimLink> );

  // Step B: LOOP on StripGeomDetUnit
  theDigiVector.reserve(10000);
  theDigiVector.clear();

  for(TrackingGeometry::DetUnitContainer::const_iterator iu = pDD->detUnits().begin(); iu != pDD->detUnits().end(); iu ++){
    if(useConfFromDB){
      //apply the cable map _before_ digitization: consider only the detis that are connected 
      if(theDetIdList.find((*iu)->geographicalId().rawId())==theDetIdList.end())
        continue;
    }
    auto sgd = dynamic_cast<StripGeomDetUnit const*>((*iu));
    if (sgd != 0){
      edm::DetSet<SiStripDigi> collectorZS((*iu)->geographicalId().rawId());
      edm::DetSet<SiStripRawDigi> collectorRaw((*iu)->geographicalId().rawId());
      edm::DetSet<StripDigiSimLink> collectorLink((*iu)->geographicalId().rawId());
      theDigiAlgo->digitize(collectorZS,collectorRaw,collectorLink,sgd,
                            gainHandle,thresholdHandle,noiseHandle,pedestalHandle, randomEngine(iEvent.streamID()));
      if(zeroSuppression){
        if(collectorZS.data.size()>0){
          theDigiVector.push_back(collectorZS);
          if( !collectorLink.data.empty() ) pOutputDigiSimLink->insert(collectorLink);
        }
      }else{
        if(collectorRaw.data.size()>0){
          theRawDigiVector.push_back(collectorRaw);
          if( !collectorLink.data.empty() ) pOutputDigiSimLink->insert(collectorLink);
        }
      }
    }
  }

  if(zeroSuppression){
    // Step C: create output collection
    std::unique_ptr<edm::DetSetVector<SiStripRawDigi> > output_virginraw(new edm::DetSetVector<SiStripRawDigi>());
    std::unique_ptr<edm::DetSetVector<SiStripRawDigi> > output_scopemode(new edm::DetSetVector<SiStripRawDigi>());
    std::unique_ptr<edm::DetSetVector<SiStripRawDigi> > output_processedraw(new edm::DetSetVector<SiStripRawDigi>());
    std::unique_ptr<edm::DetSetVector<SiStripDigi> > output(new edm::DetSetVector<SiStripDigi>(theDigiVector) );
    std::unique_ptr<std::vector<std::pair<int,std::bitset<6>>> > AffectedAPVList(new std::vector<std::pair<int,std::bitset<6>>>(theAffectedAPVvector));

    // Step D: write output to file
    iEvent.put(output, ZSDigi);
    iEvent.put(output_scopemode, SCDigi);
    iEvent.put(output_virginraw, VRDigi);
    iEvent.put(output_processedraw, PRDigi);
    if( makeDigiSimLinks_ ) iEvent.put( pOutputDigiSimLink ); // The previous EDProducer didn't name this collection so I won't either
  }else{
    // Step C: create output collection
    std::auto_ptr<edm::DetSetVector<SiStripRawDigi> > output_virginraw(new edm::DetSetVector<SiStripRawDigi>(theRawDigiVector));
    std::auto_ptr<edm::DetSetVector<SiStripRawDigi> > output_scopemode(new edm::DetSetVector<SiStripRawDigi>());
    std::auto_ptr<edm::DetSetVector<SiStripRawDigi> > output_processedraw(new edm::DetSetVector<SiStripRawDigi>());
    std::auto_ptr<edm::DetSetVector<SiStripDigi> > output(new edm::DetSetVector<SiStripDigi>() );
    // Step D: write output to file
    iEvent.put(output, ZSDigi);
    iEvent.put(output_scopemode, SCDigi);
    iEvent.put(output_virginraw, VRDigi);
    iEvent.put(output_processedraw, PRDigi);
    if( makeDigiSimLinks_ ) iEvent.put( pOutputDigiSimLink ); // The previous EDProducer didn't name this collection so I won't either
  }
}

CLHEP::HepRandomEngine* SiStripDigitizer::randomEngine(edm::StreamID const& streamID) {
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
