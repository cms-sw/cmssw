#include "Validation/MuonCSCDigis/src/CSCDigiValidation.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Validation/MuonCSCDigis/src/CSCStripDigiValidation.h"
#include "Validation/MuonCSCDigis/src/CSCWireDigiValidation.h"
#include "Validation/MuonCSCDigis/src/CSCComparatorDigiValidation.h"
#include "Validation/MuonCSCDigis/src/CSCALCTDigiValidation.h"
#include "Validation/MuonCSCDigis/src/CSCCLCTDigiValidation.h"
#include <iostream>
#include "DQMServices/Core/interface/DQMStore.h"

CSCDigiValidation::CSCDigiValidation(const edm::ParameterSet & ps)
  : dbe_( edm::Service<DQMStore>().operator->() ),
    outputFile_( ps.getParameter<std::string>("outputFile") ),
    theSimHitMap(ps.getParameter<edm::InputTag>("simHitsTag"), consumesCollector()),
    theCSCGeometry(0),
    theStripDigiValidation(0),
    theWireDigiValidation(0),
    theComparatorDigiValidation(0),
    theALCTDigiValidation(0),
    theCLCTDigiValidation(0)
{
  dbe_->setCurrentFolder("MuonCSCDigisV/CSCDigiTask");
  bool doSim = ps.getParameter<bool>("doSim");

  theStripDigiValidation = new CSCStripDigiValidation(dbe_,
                                                      ps.getParameter<edm::InputTag>("stripDigiTag"),
                                                      consumesCollector(),
                                                      doSim);
  theWireDigiValidation  = new CSCWireDigiValidation(dbe_,
                                                     ps.getParameter<edm::InputTag>("wireDigiTag"),
                                                     consumesCollector(),
                                                     doSim);
  theComparatorDigiValidation  = new CSCComparatorDigiValidation(dbe_,
                                                                 ps.getParameter<edm::InputTag>("comparatorDigiTag"),
                                                                 ps.getParameter<edm::InputTag>("stripDigiTag"),
                                                                 consumesCollector());
  theALCTDigiValidation = new CSCALCTDigiValidation(dbe_,
                                                    ps.getParameter<edm::InputTag>("alctDigiTag"),
                                                    consumesCollector());
  theCLCTDigiValidation = new CSCCLCTDigiValidation(dbe_,
                                                    ps.getParameter<edm::InputTag>("clctDigiTag"),
                                                    consumesCollector());

  if(doSim)
  {
    theStripDigiValidation->setSimHitMap(&theSimHitMap);
    theWireDigiValidation->setSimHitMap(&theSimHitMap);
    theComparatorDigiValidation->setSimHitMap(&theSimHitMap);
  }
}


CSCDigiValidation::~CSCDigiValidation()
{
  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
  delete theStripDigiValidation;
  delete theWireDigiValidation;
  delete theComparatorDigiValidation;
  delete theALCTDigiValidation;
  delete theCLCTDigiValidation;
}


void CSCDigiValidation::endJob() {
  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
}


void CSCDigiValidation::analyze(const edm::Event&e, const edm::EventSetup& eventSetup)
{
  theSimHitMap.fill(e);

  // find the geometry & conditions for this event
  edm::ESHandle<CSCGeometry> hGeom;
  eventSetup.get<MuonGeometryRecord>().get( hGeom );
  const CSCGeometry *pGeom = &*hGeom;

  theStripDigiValidation->setGeometry( pGeom );
  theWireDigiValidation->setGeometry( pGeom );
  theComparatorDigiValidation->setGeometry( pGeom );
  theALCTDigiValidation->setGeometry( pGeom );
  theCLCTDigiValidation->setGeometry( pGeom );


  theStripDigiValidation->analyze(e,eventSetup);
  theWireDigiValidation->analyze(e,eventSetup);
  theComparatorDigiValidation->analyze(e,eventSetup);
  theALCTDigiValidation->analyze(e,eventSetup);
  theCLCTDigiValidation->analyze(e,eventSetup);

}



