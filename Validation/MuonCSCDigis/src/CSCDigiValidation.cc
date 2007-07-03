#include "Validation/MuonCSCDigis/src/CSCDigiValidation.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Validation/MuonCSCDigis/src/CSCStripDigiValidation.h"
#include "Validation/MuonCSCDigis/src/CSCWireDigiValidation.h"
#include "Validation/MuonCSCDigis/src/CSCComparatorDigiValidation.h"


#include <iostream>

CSCDigiValidation::CSCDigiValidation(const edm::ParameterSet & ps)
: dbe_( edm::Service<DaqMonitorBEInterface>().operator->() ),
  outputFile_( ps.getParameter<std::string>("outputFile") ),
  theSimHitMap("MuonCSCHits"),
  theCSCGeometry(0),
  theStripDigiValidation(0),
  theWireDigiValidation(0),
  theComparatorDigiValidation(0)
{
  dbe_->setCurrentFolder("CSCDigiTask");
  bool doSim = ps.getParameter<bool>("doSim");

  theStripDigiValidation = new CSCStripDigiValidation(dbe_, ps.getParameter<edm::InputTag>("stripDigiTag"), doSim);
  theWireDigiValidation  = new CSCWireDigiValidation(dbe_, ps.getParameter<edm::InputTag>("wireDigiTag"), doSim);
  theComparatorDigiValidation  = new CSCComparatorDigiValidation(dbe_, ps.getParameter<edm::InputTag>("comparatorDigiTag"));

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

  theStripDigiValidation->analyze(e,eventSetup);
  theWireDigiValidation->analyze(e,eventSetup);
  theComparatorDigiValidation->analyze(e,eventSetup);
}



