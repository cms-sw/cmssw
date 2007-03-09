#include "Validation/CSCRecHits/src/CSCSegmentValidation.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"



CSCSegmentValidation::CSCSegmentValidation(DaqMonitorBEInterface* dbe, const edm::InputTag & inputTag)
: CSCBaseValidation(dbe, inputTag),
  theNPerEventPlot( dbe_->book1D("CSCSegmentsPerEvent", "Number of CSC segments per event", 100, 0, 100) ),
  theNRecHitsPlot( dbe_->book1D("CSCRecHitsPerSegment", "Number of CSC rec hits per segment" , 7, 0, 6) ),
  theNPerChamberTypePlot( dbe_->book1D("CSCSegmentsPerChamberType", "Number of CSC segments per chamber type", 11, 0, 10) )
{
   for(int i = 0; i < 10; ++i)
  {
    char title1[200], title2[200];
    sprintf(title1, "CSCSegmentResolution%d", i+1);
    sprintf(title2, "CSCSegmentPull%d", i+1);
    theResolutionPlots[i] = dbe_->book1D(title1, title1, 100, -1, 1);
    thePullPlots[i] = dbe_->book1D(title2, title2, 100, -1, 1);
  }

}

void CSCSegmentValidation::analyze(const edm::Event&e, const edm::EventSetup& eventSetup)
{
  // get the collection of CSCRecHsegmentItrD
  edm::Handle<CSCSegmentCollection> hRecHits;
  e.getByLabel(theInputTag, hRecHits);
  const CSCSegmentCollection * cscRecHits = hRecHits.product();

  unsigned nPerEvent = 0;

  for(CSCSegmentCollection::const_iterator segmentItr = cscRecHits->begin(); 
      segmentItr != cscRecHits->end(); segmentItr++) 
  {
    ++nPerEvent;
    int chamberType = 0;
    theNRecHitsPlot->Fill(segmentItr->nRecHits());
    theNPerChamberTypePlot->Fill(chamberType);
  }
  theNPerEventPlot->Fill(nPerEvent);

}


void CSCSegmentValidation::plotResolution(const PSimHit & simHit, const CSCSegment & segment,
                                         const CSCLayer * layer, int chamberType)
{
  GlobalPoint simHitPos = layer->toGlobal(simHit.localPosition());
  GlobalPoint segmentPos = layer->toGlobal(segment.localPosition());

  double dphi = segmentPos.phi() - simHitPos.phi();
  double rdphi = segmentPos.perp() * dphi;
  theResolutionPlots[chamberType-1]->Fill( rdphi );
  thePullPlots[chamberType-1]->Fill( rdphi/segment.localPositionError().xx() );
}

