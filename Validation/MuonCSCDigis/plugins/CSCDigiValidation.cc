#include "Validation/MuonCSCDigis/plugins/CSCDigiValidation.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Validation/MuonCSCDigis/interface/CSCALCTDigiValidation.h"
#include "Validation/MuonCSCDigis/interface/CSCCLCTDigiValidation.h"
#include "Validation/MuonCSCDigis/interface/CSCCLCTPreTriggerDigiValidation.h"
#include "Validation/MuonCSCDigis/interface/CSCCorrelatedLCTDigiValidation.h"
#include "Validation/MuonCSCDigis/interface/CSCComparatorDigiValidation.h"
#include "Validation/MuonCSCDigis/interface/CSCStripDigiValidation.h"
#include "Validation/MuonCSCDigis/interface/CSCWireDigiValidation.h"
#include "Validation/MuonCSCDigis/interface/CSCStubEfficiencyValidation.h"
#include "Validation/MuonCSCDigis/interface/CSCStubResolutionValidation.h"
#include <iostream>
#include <memory>

CSCDigiValidation::CSCDigiValidation(const edm::ParameterSet &ps)
    : doSim_(ps.getParameter<bool>("doSim")), theSimHitMap(nullptr), theCSCGeometry(nullptr) {
  // instantiatethe validation modules
  theStripDigiValidation = std::make_unique<CSCStripDigiValidation>(ps, consumesCollector());
  theWireDigiValidation = std::make_unique<CSCWireDigiValidation>(ps, consumesCollector());
  theComparatorDigiValidation = std::make_unique<CSCComparatorDigiValidation>(ps, consumesCollector());
  theALCTDigiValidation = std::make_unique<CSCALCTDigiValidation>(ps, consumesCollector());
  theCLCTDigiValidation = std::make_unique<CSCCLCTDigiValidation>(ps, consumesCollector());
  theCLCTPreTriggerDigiValidation = std::make_unique<CSCCLCTPreTriggerDigiValidation>(ps, consumesCollector());
  theCorrelatedLCTDigiValidation = std::make_unique<CSCCorrelatedLCTDigiValidation>(ps, consumesCollector());
  // set the simhit map for resolution studies
  if (doSim_) {
    theSimHitMap = new PSimHitMap(ps.getParameter<edm::InputTag>("simHitsTag"), consumesCollector());
    theStripDigiValidation->setSimHitMap(theSimHitMap);
    theWireDigiValidation->setSimHitMap(theSimHitMap);
    theComparatorDigiValidation->setSimHitMap(theSimHitMap);
    theStubEfficiencyValidation = std::make_unique<CSCStubEfficiencyValidation>(ps, consumesCollector());
    theStubResolutionValidation = std::make_unique<CSCStubResolutionValidation>(ps, consumesCollector());
  }
  geomToken_ = esConsumes<CSCGeometry, MuonGeometryRecord>();
}

CSCDigiValidation::~CSCDigiValidation() {}

void CSCDigiValidation::bookHistograms(DQMStore::IBooker &iBooker,
                                       edm::Run const &iRun,
                                       edm::EventSetup const & /* iSetup */) {
  // plot directory is set for each submodule
  theStripDigiValidation->bookHistograms(iBooker);
  theWireDigiValidation->bookHistograms(iBooker);
  theComparatorDigiValidation->bookHistograms(iBooker);
  theALCTDigiValidation->bookHistograms(iBooker);
  theCLCTDigiValidation->bookHistograms(iBooker);
  theCLCTPreTriggerDigiValidation->bookHistograms(iBooker);
  theCorrelatedLCTDigiValidation->bookHistograms(iBooker);
  if (doSim_) {
    // these plots are split over ALCT, CLCT and LCT
    theStubEfficiencyValidation->bookHistograms(iBooker);
    theStubResolutionValidation->bookHistograms(iBooker);
  }
}

void CSCDigiValidation::analyze(const edm::Event &e, const edm::EventSetup &eventSetup) {
  // find the geometry & conditions for this event
  const CSCGeometry *pGeom = &eventSetup.getData(geomToken_);

  theStripDigiValidation->setGeometry(pGeom);
  theWireDigiValidation->setGeometry(pGeom);
  theComparatorDigiValidation->setGeometry(pGeom);
  theALCTDigiValidation->setGeometry(pGeom);
  theCLCTDigiValidation->setGeometry(pGeom);
  theCLCTPreTriggerDigiValidation->setGeometry(pGeom);
  theCorrelatedLCTDigiValidation->setGeometry(pGeom);
  if (doSim_) {
    theSimHitMap->fill(e);
    theStubEfficiencyValidation->setGeometry(pGeom);
    theStubResolutionValidation->setGeometry(pGeom);
  }

  theStripDigiValidation->analyze(e, eventSetup);
  theWireDigiValidation->analyze(e, eventSetup);
  theComparatorDigiValidation->analyze(e, eventSetup);
  theALCTDigiValidation->analyze(e, eventSetup);
  theCLCTDigiValidation->analyze(e, eventSetup);
  theCLCTPreTriggerDigiValidation->analyze(e, eventSetup);
  theCorrelatedLCTDigiValidation->analyze(e, eventSetup);
  if (doSim_) {
    theStubEfficiencyValidation->analyze(e, eventSetup);
    theStubResolutionValidation->analyze(e, eventSetup);
  }
}
