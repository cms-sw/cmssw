#include "Validation/CSCRecHits/src/CSCRecHitValidation.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include <DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h>
#include "DQMServices/Core/interface/DQMStore.h"


CSCRecHitValidation::CSCRecHitValidation(const edm::ParameterSet & ps)
: theSimHitMap(ps.getParameter<edm::InputTag>("simHitsTag"), consumesCollector() ),
  theCSCGeometry(0),
  the2DValidation(0),
  theSegmentValidation(0)
{
  the2DValidation = new CSCRecHit2DValidation(
                                ps.getParameter<edm::InputTag>("recHitLabel"),
                                consumesCollector() );
  theSegmentValidation = new CSCSegmentValidation(
                                ps.getParameter<edm::InputTag>("segmentLabel"),
                                consumesCollector() );
}

CSCRecHitValidation::~CSCRecHitValidation()
{
  delete the2DValidation;
  delete theSegmentValidation;
}

void CSCRecHitValidation::bookHistograms(DQMStore::IBooker & iBooker, edm::Run const & iRun, edm::EventSetup const &)
{
  iBooker.setCurrentFolder("CSCRecHitsV/CSCRecHitTask");
  
  the2DValidation->bookHistograms(iBooker);
  theSegmentValidation->bookHistograms(iBooker);
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
