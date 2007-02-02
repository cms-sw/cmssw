#include "Validation/CSCRecHits/src/CSCRecHitValidation.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include <DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h>



CSCRecHitValidation::CSCRecHitValidation(const edm::ParameterSet & ps)
: dbe_( edm::Service<DaqMonitorBEInterface>().operator->() ),
  theInputTag( ps.getParameter<edm::InputTag>("inputObjects") ),
  theOutputFile( ps.getParameter<std::string>("outputFile") ),
  theSimHitMap("MuonCSCHits"),
  theCSCGeometry(0),
  theNPerEventPlot( dbe_->book1D("CSCRecHitsPerEvent", "Number of CSC Rec Hits per event", 100, 0, 500) )
{
   dbe_->setCurrentFolder("CSCRecHitTask");

   for(int i = 0; i < 10; ++i)
  {
    char title1[200], title2[200];
    sprintf(title1, "CSCRecHitResolution%d", i+1);
    sprintf(title2, "CSCRecHitPull%d", i+1);
    theResolutionPlots[i] = dbe_->book1D(title1, title1, 100, -1, 1);
    thePullPlots[i] = dbe_->book1D(title2, title2, 100, -1, 1);
  }

}


CSCRecHitValidation::~CSCRecHitValidation()
{
  if ( theOutputFile.size() != 0 && dbe_ ) dbe_->save(theOutputFile);
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

  // get the collection of CSCRecHrecHitItrD
  edm::Handle<CSCRecHit2DCollection> hRecHits;
  e.getByLabel(theInputTag, hRecHits);
  const CSCRecHit2DCollection * cscRecHits = hRecHits.product();

  unsigned nPerEvent = 0;

  for(CSCRecHit2DCollection::const_iterator recHitItr = cscRecHits->begin(); 
      recHitItr != cscRecHits->end(); recHitItr++) 
  {
    ++nPerEvent;
    int detId = (*recHitItr).cscDetId().rawId();
    edm::PSimHitContainer simHits = theSimHitMap.hits(detId);

    if(simHits.size() == 1)
    {
      const CSCLayer * layer = findLayer(detId);
      int chamberType = layer->chamber()->specs()->chamberType();
      plotResolution(simHits[0], *recHitItr, layer, chamberType);
    }

  }    
}


void CSCRecHitValidation::plotResolution(const PSimHit & simHit, const CSCRecHit2D & recHit,
                                         const CSCLayer * layer, int chamberType)
{
  GlobalPoint simHitPos = layer->toGlobal(simHit.localPosition());
  GlobalPoint recHitPos = layer->toGlobal(recHit.localPosition());

  double dphi = recHitPos.phi() - simHitPos.phi();
  double rdphi = recHitPos.perp() * dphi;
  theResolutionPlots[chamberType-1]->Fill( rdphi );
  thePullPlots[chamberType-1]->Fill( rdphi/recHit.localPositionError().xx() );
}


const CSCLayer * CSCRecHitValidation::findLayer(int detId) const {
  assert(theCSCGeometry != 0);
  const GeomDetUnit* detUnit = theCSCGeometry->idToDetUnit(CSCDetId(detId));
  return dynamic_cast<const CSCLayer *>(detUnit);
}

