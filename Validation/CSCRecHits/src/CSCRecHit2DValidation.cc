#include "Validation/CSCRecHits/src/CSCRecHit2DValidation.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/DQMStore.h"



CSCRecHit2DValidation::CSCRecHit2DValidation(DQMStore* dbe, const edm::InputTag & inputTag, edm::ConsumesCollector && iC)
: CSCBaseValidation(dbe, inputTag),
  theNPerEventPlot( dbe_->book1D("CSCRecHitsPerEvent", "Number of CSC Rec Hits per event", 100, 0, 500) )
{
   rechits_Token_ = iC.consumes<CSCRecHit2DCollection>(inputTag);

   dbe_->setCurrentFolder("CSCRecHitsV/CSCRecHitTask");

   for(int i = 0; i < 10; ++i)
  {
    char title1[200], title2[200], title3[200], title4[200], title5[200], title6[200], title7[200], title8[200], title9[200];
    sprintf(title1, "CSCRecHitResolution%d", i+1);
    sprintf(title2, "CSCRecHitPull%d", i+1);
    sprintf(title3, "CSCRecHitYResolution%d", i+1);
    sprintf(title4, "CSCRecHitYPull%d", i+1);
    sprintf(title5, "CSCRecHitPosInStrip%d", i+1);
    sprintf(title6, "CSCSimHitPosInStrip%d", i+1);
    sprintf(title7, "CSCRecHit%d", i+1);
    sprintf(title8, "CSCSimHit%d", i+1);
    sprintf(title9, "CSCTPeak%d", i+1);


    theResolutionPlots[i] = dbe_->book1D(title1, title1, 100, -0.2, 0.2);
    thePullPlots[i] = dbe_->book1D(title2, title2, 100, -3, 3);
    theYResolutionPlots[i] = dbe_->book1D(title3, title3, 100, -5, 5);
    theYPullPlots[i] = dbe_->book1D(title4, title4, 100, -3, 3);
    theRecHitPosInStrip[i] = dbe_->book1D(title5, title5, 100, -2, 2);
    theSimHitPosInStrip[i] = dbe_->book1D(title6, title6, 100, -2, 2);
    theScatterPlots[i] = dbe->book2D(title7, title7, 200, -20, 20, 200, -250, 250);
    theSimHitScatterPlots[i] = dbe->book2D(title8, title8, 200, -20, 20, 200, -250, 250);
    theTPeaks[i] =  dbe->book1D(title9, title9, 200, 0, 400);
  }

}

CSCRecHit2DValidation::~CSCRecHit2DValidation()
{
  for(int i = 0; i < 10; ++i)
  {
    edm::LogInfo("CSCRecHitValidation") << "Resolution of " << theResolutionPlots[i]->getName() << " is " << theResolutionPlots[i]->getRMS();
    edm::LogInfo("CSCRecHitValidation") << "Peak Time is " << theTPeaks[i]->getMean();
  }
}


void CSCRecHit2DValidation::analyze(const edm::Event&e, const edm::EventSetup& eventSetup)
{
  // get the collection of CSCRecHrecHitItrD
  edm::Handle<CSCRecHit2DCollection> hRecHits;
  e.getByToken(rechits_Token_, hRecHits);
  const CSCRecHit2DCollection * cscRecHits = hRecHits.product();

  unsigned nPerEvent = 0;

  for(CSCRecHit2DCollection::const_iterator recHitItr = cscRecHits->begin(); 
      recHitItr != cscRecHits->end(); recHitItr++) 
  {
    ++nPerEvent;
    int detId = (*recHitItr).cscDetId().rawId();
    edm::PSimHitContainer simHits = theSimHitMap->hits(detId);
    const CSCLayer * layer = findLayer(detId);
    int chamberType = layer->chamber()->specs()->chamberType();
    theTPeaks[chamberType-1]->Fill(recHitItr->tpeak());
    if(simHits.size() == 1)
    {
      plotResolution(simHits[0], *recHitItr, layer, chamberType);
    }
    float localX = recHitItr->localPosition().x();
    float localY = recHitItr->localPosition().y();
    //theYPlots[chamberType-1]->Fill(localY);
    // find a local phi
    float globalR = layer->toGlobal(LocalPoint(0.,0.,0.)).perp();
    GlobalPoint axisThruChamber(globalR+localY, localX, 0.);
    float localPhi = axisThruChamber.phi().degrees();
    //thePhiPlots[chamberType-1]->Fill(axisThruChamber.phi().degrees());
    theScatterPlots[chamberType-1]->Fill( localPhi, localY);
  }    
  theNPerEventPlot->Fill(nPerEvent);
return;
  // fill sim hits
  std::vector<int> layersWithSimHits = theSimHitMap->detsWithHits();
  for(unsigned i = 0; i < layersWithSimHits.size(); ++i)
   {
    edm::PSimHitContainer simHits = theSimHitMap->hits(layersWithSimHits[i]);
    for(edm::PSimHitContainer::const_iterator hitItr = simHits.begin(); hitItr != simHits.end(); ++hitItr)
    {
    const CSCLayer * layer = findLayer(layersWithSimHits[i]);
    int chamberType = layer->chamber()->specs()->chamberType();
      float localX = hitItr->localPosition().x();
      float localY = hitItr->localPosition().y();
      //theYPlots[chamberType-1]->Fill(localY);
      // find a local phi
      float globalR = layer->toGlobal(LocalPoint(0.,0.,0.)).perp();
      GlobalPoint axisThruChamber(globalR+localY, localX, 0.);
      float localPhi = axisThruChamber.phi().degrees();
      //thePhiPlots[chamberType-1]->Fill(axisThruChamber.phi().degrees());
      theSimHitScatterPlots[chamberType-1]->Fill( localPhi, localY);
    }
  }

}


void CSCRecHit2DValidation::plotResolution(const PSimHit & simHit, const CSCRecHit2D & recHit,
                                         const CSCLayer * layer, int chamberType)
{
  GlobalPoint simHitPos = layer->toGlobal(simHit.localPosition());
  GlobalPoint recHitPos = layer->toGlobal(recHit.localPosition());

  double dphi = recHitPos.phi() - simHitPos.phi();
  double rdphi = recHitPos.perp() * dphi;
  theResolutionPlots[chamberType-1]->Fill( rdphi );
  thePullPlots[chamberType-1]->Fill( rdphi/ sqrt(recHit.localPositionError().xx()) );
  double dy = recHit.localPosition().y() - simHit.localPosition().y();
  theYResolutionPlots[chamberType-1]->Fill( dy );
  theYPullPlots[chamberType-1]->Fill( dy/ sqrt(recHit.localPositionError().yy()) );

  const CSCLayerGeometry * layerGeometry = layer->geometry();
  float recStrip = layerGeometry->strip(recHit.localPosition());
  float simStrip = layerGeometry->strip(simHit.localPosition());
  theRecHitPosInStrip[chamberType-1]->Fill( recStrip - int(recStrip) );
  theSimHitPosInStrip[chamberType-1]->Fill( simStrip - int(simStrip) );
}

