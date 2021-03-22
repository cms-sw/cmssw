#include "Validation/MuonCSCDigis/plugins/CSCDigiValidation.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Validation/MuonCSCDigis/interface/CSCALCTDigiValidation.h"
#include "Validation/MuonCSCDigis/interface/CSCCLCTDigiValidation.h"
#include "Validation/MuonCSCDigis/interface/CSCComparatorDigiValidation.h"
#include "Validation/MuonCSCDigis/interface/CSCStripDigiValidation.h"
#include "Validation/MuonCSCDigis/interface/CSCWireDigiValidation.h"
#include "Validation/MuonCSCDigis/interface/CSCStubEfficiencyValidation.h"
#include <iostream>
#include <memory>

CSCDigiValidation::CSCDigiValidation(const edm::ParameterSet &ps)
    : doSim_(ps.getParameter<bool>("doSim")),
      theSimHitMap(ps.getParameter<edm::InputTag>("simHitsTag"), consumesCollector()),
      theCSCGeometry(nullptr),
      theStripDigiValidation(nullptr),
      theWireDigiValidation(nullptr),
      theComparatorDigiValidation(nullptr),
      theALCTDigiValidation(nullptr),
      theCLCTDigiValidation(nullptr),
      theStubEfficiencyValidation(nullptr) {
  // instantiatethe validation modules
  theStripDigiValidation = std::make_unique<CSCStripDigiValidation>(ps, consumesCollector());
  theWireDigiValidation = std::make_unique<CSCWireDigiValidation>(ps, consumesCollector());
  theComparatorDigiValidation = std::make_unique<CSCComparatorDigiValidation>(ps, consumesCollector());
  theALCTDigiValidation = std::make_unique<CSCALCTDigiValidation>(ps, consumesCollector());
  theCLCTDigiValidation = std::make_unique<CSCCLCTDigiValidation>(ps, consumesCollector());
  theStubEfficiencyValidation = std::make_unique<CSCStubEfficiencyValidation>(ps, consumesCollector());

  // set the simhit map for resolution studies
  if (doSim_) {
    theStripDigiValidation->setSimHitMap(&theSimHitMap);
    theWireDigiValidation->setSimHitMap(&theSimHitMap);
    theComparatorDigiValidation->setSimHitMap(&theSimHitMap);
  }
  geomToken_ = esConsumes<CSCGeometry, MuonGeometryRecord>();
}

CSCDigiValidation::~CSCDigiValidation() {}

void CSCDigiValidation::bookHistograms(DQMStore::IBooker &iBooker,
                                       edm::Run const &iRun,
                                       edm::EventSetup const & /* iSetup */) {
  iBooker.setCurrentFolder("MuonCSCDigisV/CSCDigiTask");
  theStripDigiValidation->bookHistograms(iBooker);
  theWireDigiValidation->bookHistograms(iBooker);
  theComparatorDigiValidation->bookHistograms(iBooker);
  theALCTDigiValidation->bookHistograms(iBooker);
  theCLCTDigiValidation->bookHistograms(iBooker);
  theStubEfficiencyValidation->bookHistograms(iBooker);
}

void CSCDigiValidation::analyze(const edm::Event &e, const edm::EventSetup &eventSetup) {
  theSimHitMap.fill(e);

  // find the geometry & conditions for this event
  const CSCGeometry *pGeom = &eventSetup.getData(geomToken_);

  theStripDigiValidation->setGeometry(pGeom);
  theWireDigiValidation->setGeometry(pGeom);
  theComparatorDigiValidation->setGeometry(pGeom);
  theALCTDigiValidation->setGeometry(pGeom);
  theCLCTDigiValidation->setGeometry(pGeom);
  theStubEfficiencyValidation->setGeometry(pGeom);

  theStripDigiValidation->analyze(e, eventSetup);
  theWireDigiValidation->analyze(e, eventSetup);
  theComparatorDigiValidation->analyze(e, eventSetup);
  theALCTDigiValidation->analyze(e, eventSetup);
  theCLCTDigiValidation->analyze(e, eventSetup);
  theStubEfficiencyValidation->analyze(e, eventSetup);
}
