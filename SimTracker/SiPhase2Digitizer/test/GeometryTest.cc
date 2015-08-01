// -*- C++ -*-
//
// Package:    SimHitTest
// Class:      SimHitTest
// 
/**\class GeometryTest GeometryTest.cc 

 Description: Access Geometry and checks ttype thickness etc. 

*/
//
// Author:  Suchandra Dutta
// Created:  July 2015
//
//
// system include files
#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"

using namespace std;

class GeometryTest : public edm::EDAnalyzer {

public:

  explicit GeometryTest(const edm::ParameterSet&);
  ~GeometryTest();
  virtual void beginJob();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob(); 
  void accumulateHits(edm::Handle<std::vector<PSimHit> > hSimHits, const TrackerGeometry& theTracker);
  
private:
  const std::string hitsProducer_;
  const std::vector<std::string> trackerContainers_;
  const std::string geometryType_;
  std::map<unsigned int, GeomDetUnit*> detectorUnits_;

public:
};
//
// constructors and destructor
//
GeometryTest::GeometryTest(const edm::ParameterSet& iConfig) :
  hitsProducer_(iConfig.getParameter<std::string>("hitsProducer")),
  trackerContainers_(iConfig.getParameter<std::vector<std::string> >("ROUList")),
  geometryType_(iConfig.getParameter<std::string>("GeometryType"))
{
}
GeometryTest::~GeometryTest() {
}

void GeometryTest::beginJob() {
}

void GeometryTest::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::ESHandle<TrackerGeometry> pDD;
  iSetup.get<TrackerDigiGeometryRecord>().get(geometryType_, pDD);
  const TrackerGeometry& theTracker(*pDD);
  // the DetUnits
  TrackingGeometry::DetContainer theDetUnits = theTracker.dets();
  std::cout <<  " Geometry node for TrackingGeometry is  " << &(*pDD)
	    << "\n I have " << pDD->dets().size() << " detectors "
	    << "\n I have " << pDD->detTypes().size() << " types "
	    << "\n theDetUnits has " << theDetUnits.size() << " dets "
	    << std::endl;
  if (true) { // Replace with ESWatcher 
    std::cout << ">>> detectorUnits:" << std::endl;
    detectorUnits_.clear();
    for (auto iu = pDD->detUnits().begin(); iu != pDD->detUnits().end(); ++iu) {
      unsigned int detId_raw = (*iu)->geographicalId().rawId();
      DetId detId = DetId(detId_raw);
      if (detId.det() == DetId::Detector::Tracker) {
	std::cout << "\tdetId_raw: " << detId_raw 
		  << ", det: " << detId.det() 
		  << ", subdet: " << detId.subdetId()
                  << ", Detector Type: " << theTracker.getDetectorType(detId_raw)
                  << ", Detector Thickness: " << theTracker.getDetectorThickness(detId_raw)
		  << std::endl;
        
	//	PixelGeomDetUnit* pixdet = dynamic_cast<PixelGeomDetUnit*>(*iu);
	//	assert(pixdet);
	detectorUnits_.insert({detId_raw, (*iu)});
      }
    }
    std::cout  << "\tdetectorUnits size:" << detectorUnits_.size() << std::endl;
  }

  for (auto it = trackerContainers_.begin(); it != trackerContainers_.end(); ++it) {
    edm::Handle<std::vector<PSimHit> > simHits;
    edm::InputTag tag(hitsProducer_, *it);
    std::cout << ">>> tag: " << tag << std::endl;
    std::cout << std::flush;
    
    iEvent.getByLabel(tag, simHits);
    if (simHits.isValid()) accumulateHits(simHits, theTracker);
  }
}
void GeometryTest::accumulateHits(edm::Handle<std::vector<PSimHit> > hSimHits, const TrackerGeometry& theTracker) {
  std::set<unsigned int> detIds;
  std::vector<PSimHit> const& simHits = *hSimHits.product();
  for (auto& hit: simHits) {
    unsigned int detId_raw = hit.detUnitId();
    DetId detId(detId_raw);
    if (detIds.insert(detId_raw).second) {
      // The insert succeeded, so this detector element has not yet been processed.
      if (detectorUnits_.find(detId_raw) == detectorUnits_.end()) {
	std::cout << ">>> accumulateHits: not found in map! detId_raw: " << detId_raw 
		  << ", det: " << detId.det() 
		  << ", subdet: " << detId.subdetId()
		  << std::endl;
	//const auto* gdet = theTracker.idToDet(detId); 
	//if (gdet) {
	//  std::cout << "theTracker.idToDet(detId) succeeds!";
	//}  
	continue; 
      }  
      std::cout << "\t>>> accumulateHits: detId found in map! detid: " << detId_raw 
		<< ", det: " << detId.det() 
		<< ", subdet: " << detId.subdetId()
		<< std::endl;
    }
  }
}
void GeometryTest::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(GeometryTest);
