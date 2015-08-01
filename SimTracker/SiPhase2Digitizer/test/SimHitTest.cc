// -*- C++ -*-
//
// Package:    SimHitTest
// Class:      SimHitTest
// 
/**\class SimHitTest SimHitTest.cc 

 Description: Access SimHit Collection and make a few plots

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

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

// DQM Histograming
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

class SimHitTest : public edm::EDAnalyzer {

public:

  explicit SimHitTest(const edm::ParameterSet&);
  ~SimHitTest();
  virtual void beginJob();
  virtual void beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup);
  virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);

  virtual void endJob(); 

  

private:


  void bookHistos();

  MonitorElement*   SimulatedHitPosRZ;
  MonitorElement*   SimulatedHitPosXY;

  DQMStore* dqmStore_;
  edm::ParameterSet config_;
};

// // constructors 
//
SimHitTest::SimHitTest(const edm::ParameterSet& iConfig) :
  dqmStore_(edm::Service<DQMStore>().operator->()),
  config_(iConfig)
{
  edm::LogInfo("SimHitTest") << ">>> Construct SimHitTest ";
}

//
// destructor
//
SimHitTest::~SimHitTest() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  edm::LogInfo("SimHitTest")<< ">>> Destroy SimHitTest ";
}
//
// -- Begin Job
//
void SimHitTest::beginJob() {
   edm::LogInfo("SimHitTest")<< "Initialize SimHitTest ";
}
//
// -- Begin Run
//
void SimHitTest::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup){
  bookHistos();
}
//
// -- Analyze
//
void SimHitTest::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  
  // Get geometry information
  edm::ESHandle<TrackerGeometry> tracker;
  iSetup.get<TrackerDigiGeometryRecord>().get( tracker );

  // PSimHits
  edm::Handle<edm::PSimHitContainer> simHits;
  iEvent.getByLabel("g4SimHits","TrackerHitsPixelBarrelLowTof" ,simHits);
  for(std::vector<PSimHit>::const_iterator isim = simHits->begin(); isim != simHits->end(); ++isim){
    DetId detid=DetId(isim->detUnitId());
    std::cout << " DetId " << detid << std::endl;
    /*    const GeomDetUnit * det=(const GeomDetUnit*)tracker->idToDetUnit( detid );
    GlobalPoint gpos=det->toGlobal(isim->localPosition());
    SimulatedHitPosRZ->Fill(gpos.z()*10.0, gpos.mag()*10.0);   
    SimulatedHitPosXY->Fill(gpos.x()*10.0, gpos.y()*10.0);   */
  }
}
//
// -- Book Histograms
//
  void SimHitTest::bookHistos() {
  std::string top_folder = config_.getParameter<std::string>("TopFolderName");
  std::stringstream folder_name;

  dqmStore_->cd();
  folder_name << top_folder << "/" << "SimHitInfo" ;
  dqmStore_->setCurrentFolder(folder_name.str());
  edm::LogInfo("SimHitTest")<< " Booking Histograms in : " << folder_name.str();
  
  SimulatedHitPosRZ = dqmStore_->book2D("SimHitPosRZ", "Simulated Hits in RZ ", 1500, 0.0, 3000.0, 300, 0.0, 600.0);
  SimulatedHitPosXY = dqmStore_->book2D("SimHitPosXY", "Simulated Hits in XY ", 600, 0.0, 600.0, 600, 0.0, 600.0);

}
//
// -- End Job
//
void SimHitTest::endJob(){
  dqmStore_->cd();
  dqmStore_->showDirStructure();  
}
//define this as a plug-in
DEFINE_FWK_MODULE(SimHitTest);
