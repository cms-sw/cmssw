#include "Validation/CSCRecHits/src/CSCRecHitValidation.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include <DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h>
#include "DQMServices/Core/interface/DQMStore.h"



CSCRecHitValidation::CSCRecHitValidation(const edm::ParameterSet & ps)
: dbe_( edm::Service<DQMStore>().operator->() ),
  theOutputFile( ps.getParameter<std::string>("outputFile") ),
  theSimHitMap(ps.getParameter<edm::InputTag>("simHitsTag")),
  theCSCGeometry(0),
  the2DValidation(0),
  theSegmentValidation(0)
{
  the2DValidation = new CSCRecHit2DValidation(dbe_, ps.getParameter<edm::InputTag>("recHitLabel"), consumesCollector() );
  theSegmentValidation = new CSCSegmentValidation(dbe_, ps.getParameter<edm::InputTag>("segmentLabel"), consumesCollector() );
}


CSCRecHitValidation::~CSCRecHitValidation()
{
  if ( theOutputFile.size() != 0 && dbe_ ) dbe_->save(theOutputFile);
  delete the2DValidation;
  delete theSegmentValidation;
}


void CSCRecHitValidation::endJob() {
  if ( theOutputFile.size() != 0 && dbe_ ) dbe_->save(theOutputFile);
}


void CSCRecHitValidation::analyze(const edm::Event&e, const edm::EventSetup& eventSetup)
{
  theSimHitMap.fill(e);

  // find the geometry & conditions for this event
  edm::ESHandle<CSCGeometry> hGeom;
  eventSetup.get<MuonGeometryRecord>().get( hGeom );
  theCSCGeometry = &*hGeom;

  the2DValidation->setGeometry(theCSCGeometry);
  the2DValidation->setSimHitMap(&theSimHitMap);

  theSegmentValidation->setGeometry(theCSCGeometry);
  theSegmentValidation->setSimHitMap(&theSimHitMap);

  the2DValidation->analyze(e, eventSetup);
  theSegmentValidation->analyze(e, eventSetup);

}
