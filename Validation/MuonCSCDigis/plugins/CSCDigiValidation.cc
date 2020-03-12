#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Validation/MuonCSCDigis/interface/CSCALCTDigiValidation.h"
#include "Validation/MuonCSCDigis/interface/CSCCLCTDigiValidation.h"
#include "Validation/MuonCSCDigis/interface/CSCComparatorDigiValidation.h"
#include "Validation/MuonCSCDigis/plugins/CSCDigiValidation.h"
#include "Validation/MuonCSCDigis/interface/CSCStripDigiValidation.h"
#include "Validation/MuonCSCDigis/interface/CSCWireDigiValidation.h"
#include <iostream>

CSCDigiValidation::CSCDigiValidation(const edm::ParameterSet &ps)
    : doSim_(ps.getParameter<bool>("doSim")),
      theSimHitMap(ps.getParameter<edm::InputTag>("simHitsTag"), consumesCollector()),
      theCSCGeometry(nullptr),
      theStripDigiValidation(nullptr),
      theWireDigiValidation(nullptr),
      theComparatorDigiValidation(nullptr),
      theALCTDigiValidation(nullptr),
      theCLCTDigiValidation(nullptr) {
  theStripDigiValidation.reset(
      new CSCStripDigiValidation(ps.getParameter<edm::InputTag>("stripDigiTag"), consumesCollector()));
  theWireDigiValidation.reset(
      new CSCWireDigiValidation(ps.getParameter<edm::InputTag>("wireDigiTag"), consumesCollector(), doSim_));
  theComparatorDigiValidation.reset(new CSCComparatorDigiValidation(ps.getParameter<edm::InputTag>("comparatorDigiTag"),
                                                                    ps.getParameter<edm::InputTag>("stripDigiTag"),
                                                                    consumesCollector()));
  theALCTDigiValidation.reset(
      new CSCALCTDigiValidation(ps.getParameter<edm::InputTag>("alctDigiTag"), consumesCollector()));
  theCLCTDigiValidation.reset(
      new CSCCLCTDigiValidation(ps.getParameter<edm::InputTag>("clctDigiTag"), consumesCollector()));

  if (doSim_) {
    theStripDigiValidation->setSimHitMap(&theSimHitMap);
    theWireDigiValidation->setSimHitMap(&theSimHitMap);
    theComparatorDigiValidation->setSimHitMap(&theSimHitMap);
  }
}

CSCDigiValidation::~CSCDigiValidation() {}

void CSCDigiValidation::bookHistograms(DQMStore::IBooker &iBooker,
                                       edm::Run const &iRun,
                                       edm::EventSetup const & /* iSetup */) {
  iBooker.setCurrentFolder("MuonCSCDigisV/CSCDigiTask");
  theStripDigiValidation->bookHistograms(iBooker, doSim_);
  theWireDigiValidation->bookHistograms(iBooker);
  theComparatorDigiValidation->bookHistograms(iBooker);
  theALCTDigiValidation->bookHistograms(iBooker);
  theCLCTDigiValidation->bookHistograms(iBooker);
}

void CSCDigiValidation::analyze(const edm::Event &e, const edm::EventSetup &eventSetup) {
  theSimHitMap.fill(e);

  // find the geometry & conditions for this event
  edm::ESHandle<CSCGeometry> hGeom;
  eventSetup.get<MuonGeometryRecord>().get(hGeom);
  const CSCGeometry *pGeom = &*hGeom;

  theStripDigiValidation->setGeometry(pGeom);
  theWireDigiValidation->setGeometry(pGeom);
  theComparatorDigiValidation->setGeometry(pGeom);
  theALCTDigiValidation->setGeometry(pGeom);
  theCLCTDigiValidation->setGeometry(pGeom);

  theStripDigiValidation->analyze(e, eventSetup);
  theWireDigiValidation->analyze(e, eventSetup);
  theComparatorDigiValidation->analyze(e, eventSetup);
  theALCTDigiValidation->analyze(e, eventSetup);
  theCLCTDigiValidation->analyze(e, eventSetup);
}
