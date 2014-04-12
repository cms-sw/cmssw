#include "Validation/CSCRecHits/src/CSCSegmentValidation.h"
#include <algorithm>
#include "DQMServices/Core/interface/DQMStore.h"



CSCSegmentValidation::CSCSegmentValidation(DQMStore* dbe, const edm::InputTag & inputTag, edm::ConsumesCollector && iC)
: CSCBaseValidation(dbe, inputTag),
  theLayerHitsPerChamber(),
  theChamberSegmentMap(),
  theShowerThreshold(10),
  theNPerEventPlot( dbe_->book1D("CSCSegmentsPerEvent", "Number of CSC segments per event", 100, 0, 50) ),
  theNRecHitsPlot( dbe_->book1D("CSCRecHitsPerSegment", "Number of CSC rec hits per segment" , 8, 0, 7) ),
  theNPerChamberTypePlot( dbe_->book1D("CSCSegmentsPerChamberType", "Number of CSC segments per chamber type", 11, 0, 10) ),
  theTypePlot4HitsNoShower( dbe_->book1D("CSCSegments4HitsNoShower", "", 100, 0, 10) ),
  theTypePlot4HitsNoShowerSeg( dbe_->book1D("CSCSegments4HitsNoShowerSeg", "", 100, 0, 10) ),
  theTypePlot4HitsShower( dbe_->book1D("CSCSegments4HitsShower", "", 100, 0, 10) ),
  theTypePlot4HitsShowerSeg( dbe_->book1D("CSCSegments4HitsShowerSeg", "", 100, 0, 10) ),
  theTypePlot5HitsNoShower( dbe_->book1D("CSCSegments5HitsNoShower", "", 100, 0, 10) ),
  theTypePlot5HitsNoShowerSeg( dbe_->book1D("CSCSegments5HitsNoShowerSeg", "", 100, 0, 10) ),
  theTypePlot5HitsShower( dbe_->book1D("CSCSegments5HitsShower", "", 100, 0, 10) ),
  theTypePlot5HitsShowerSeg( dbe_->book1D("CSCSegments5HitsShowerSeg", "", 100, 0, 10) ),
  theTypePlot6HitsNoShower( dbe_->book1D("CSCSegments6HitsNoShower", "", 100, 0, 10) ),
  theTypePlot6HitsNoShowerSeg( dbe_->book1D("CSCSegments6HitsNoShowerSeg", "", 100, 0, 10) ),
  theTypePlot6HitsShower( dbe_->book1D("CSCSegments6HitsShower", "", 100, 0, 10) ),
  theTypePlot6HitsShowerSeg( dbe_->book1D("CSCSegments6HitsShowerSeg", "", 100, 0, 10) )
{
   segments_Token_ = iC.consumes<CSCSegmentCollection>(inputTag);

   dbe_->setCurrentFolder("CSCRecHitsV/CSCRecHitTask");
   for(int i = 0; i < 10; ++i)
  {
    char title1[200], title2[200], title3[200], title4[200],  
         title5[200], title6[200], title7[200], title8[200];
    sprintf(title1, "CSCSegmentRdPhiResolution%d", i+1);
    sprintf(title2, "CSCSegmentRdPhiPull%d", i+1);
    sprintf(title3, "CSCSegmentThetaResolution%d", i+1);
    sprintf(title4, "CSCSegmentThetaPull%d", i+1);
    sprintf(title5, "CSCSegmentdXdZResolution%d", i+1);
    sprintf(title6, "CSCSegmentdXdZPull%d", i+1);
    sprintf(title7, "CSCSegmentdYdZResolution%d", i+1);
    sprintf(title8, "CSCSegmentdYdZPull%d", i+1);


    theRdPhiResolutionPlots[i] = dbe_->book1D(title1, title1, 100, -0.4, 0.4);
    theRdPhiPullPlots[i] = dbe_->book1D(title2, title2, 100, -5, 5);
    theThetaResolutionPlots[i] = dbe_->book1D(title3, title3, 100, -1, 1);
    theThetaPullPlots[i] = dbe_->book1D(title4, title4, 100, -5, 5);
    thedXdZResolutionPlots[i] = dbe_->book1D(title5, title5, 100, -1, 1);
    thedXdZPullPlots[i] = dbe_->book1D(title6, title6, 100, -5, 5);
    thedYdZResolutionPlots[i] = dbe_->book1D(title7, title7, 100, -1, 1);
    thedYdZPullPlots[i] = dbe_->book1D(title8, title8, 100, -5, 5);

  }
}

void CSCSegmentValidation::analyze(const edm::Event&e, const edm::EventSetup& eventSetup)
{
  // get the collection of CSCRecHsegmentItrD
  edm::Handle<CSCSegmentCollection> hRecHits;
  e.getByToken(segments_Token_, hRecHits);
  const CSCSegmentCollection * cscRecHits = hRecHits.product();

  theChamberSegmentMap.clear();
  unsigned nPerEvent = 0;
  for(CSCSegmentCollection::const_iterator segmentItr = cscRecHits->begin(); 
      segmentItr != cscRecHits->end(); segmentItr++) 
  {
    ++nPerEvent;
    int detId = segmentItr->geographicalId().rawId();
    int chamberType = whatChamberType(detId);

    theNRecHitsPlot->Fill(segmentItr->nRecHits());
    theNPerChamberTypePlot->Fill(chamberType);
    theChamberSegmentMap[detId].push_back(*segmentItr);

    // do the resolution plots
    const PSimHit * hit = keyHit(detId);
    if(hit != 0) 
    {
      const CSCLayer * layer = findLayer(hit->detUnitId());
      plotResolution(*hit, *segmentItr, layer, chamberType);
    }  
  }

  theNPerEventPlot->Fill(nPerEvent);

  fillLayerHitsPerChamber();
  fillEfficiencyPlots();
}


void CSCSegmentValidation::fillEfficiencyPlots()
{
    // now plot efficiency by looping over all chambers with hits
  for(ChamberHitMap::const_iterator mapItr = theLayerHitsPerChamber.begin(),
      mapEnd = theLayerHitsPerChamber.end();
      mapItr != mapEnd;
      ++mapItr)
  {
    int chamberId = mapItr->first;
    int nHitsInChamber = mapItr->second.size();
    bool isShower = (nHitsInChamber > theShowerThreshold);
    bool hasSeg = hasSegment(chamberId);
    int chamberType = whatChamberType(chamberId);
    // find how many layers were hit in this chamber
    std::vector<int> v = mapItr->second;
    std::sort(v.begin(), v.end());
    // maybe can just count
    v.erase(std::unique(v.begin(), v.end()), v.end());
    int nLayersHit = v.size();

    if(nLayersHit == 4)
    {

      if(isShower) theTypePlot4HitsShower->Fill(chamberType);
      else         theTypePlot4HitsNoShower->Fill(chamberType);

      if(hasSeg) 
      {
        if(isShower) theTypePlot4HitsShowerSeg->Fill(chamberType);
        else         theTypePlot4HitsNoShowerSeg->Fill(chamberType);
      } 
    }

    if(nLayersHit == 5)
    {

      if(isShower) theTypePlot5HitsShower->Fill(chamberType);
      else         theTypePlot5HitsNoShower->Fill(chamberType);

      if(hasSeg)
      {
        if(isShower) theTypePlot5HitsShowerSeg->Fill(chamberType);
        else         theTypePlot5HitsNoShowerSeg->Fill(chamberType);
      }
    }

    if(nLayersHit == 6)
    {

      if(isShower) theTypePlot6HitsShower->Fill(chamberType);
      else         theTypePlot6HitsNoShower->Fill(chamberType);

      if(hasSeg)
      {
        if(isShower) theTypePlot6HitsShowerSeg->Fill(chamberType);
        else         theTypePlot6HitsNoShowerSeg->Fill(chamberType);
      }
    }


  }
}

bool CSCSegmentValidation::hasSegment(int chamberId) const
{
  return (theChamberSegmentMap.find(chamberId) != theChamberSegmentMap.end());
}


int CSCSegmentValidation::whatChamberType(int detId)
{
  CSCDetId cscDetId(detId);
  return CSCChamberSpecs::whatChamberType(cscDetId.station(), cscDetId.ring());
}


void CSCSegmentValidation::plotResolution(const PSimHit & simHit, const CSCSegment & segment,
                                         const CSCLayer * layer, int chamberType)
{
  GlobalPoint simHitPos = layer->toGlobal(simHit.localPosition());
  GlobalPoint segmentPos = layer->toGlobal(segment.localPosition());
  LocalVector simHitDir = simHit.localDirection();
  LocalVector segmentDir = segment.localDirection();

  double dphi = segmentPos.phi() - simHitPos.phi();
  double rdphi = segmentPos.perp() * dphi;
  double dtheta = segmentPos.theta() - simHitPos.theta();

  double sigmax = sqrt(segment.localPositionError().xx());
  //double sigmay = sqrt(segment.localPositionError().yy());

  double ddxdz = segmentDir.x()/segmentDir.z() - simHitDir.x()/simHitDir.z();
  double ddydz = segmentDir.y()/segmentDir.z() - simHitDir.y()/simHitDir.z();
  double sigmadxdz = sqrt(segment.localDirectionError().xx());
  double sigmadydz = sqrt(segment.localDirectionError().yy()); 

  theRdPhiResolutionPlots[chamberType-1]->Fill( rdphi );
  theRdPhiPullPlots[chamberType-1]->Fill( rdphi/sigmax );
  theThetaResolutionPlots[chamberType-1]->Fill( dtheta );
  //theThetaPullPlots[chamberType-1]->Fill( dy/sigmay );

  thedXdZResolutionPlots[chamberType-1]->Fill( ddxdz );
  thedXdZPullPlots[chamberType-1]->Fill( ddxdz/sigmadxdz );
  thedYdZResolutionPlots[chamberType-1]->Fill( ddydz );
  thedYdZPullPlots[chamberType-1]->Fill( ddydz/sigmadydz );
}


void CSCSegmentValidation::fillLayerHitsPerChamber()
{
  theLayerHitsPerChamber.clear();
  std::vector<int> layersHit = theSimHitMap->detsWithHits();
  for(std::vector<int>::const_iterator layerItr = layersHit.begin(), 
      layersHitEnd = layersHit.end();
      layerItr != layersHitEnd;
      ++layerItr)
  {
    CSCDetId layerId(*layerItr);
    CSCDetId chamberId = layerId.chamberId();
    int nhits = theSimHitMap->hits(*layerItr).size();
    // multiple entries, so we can see showers
    for(int i = 0; i < nhits; ++i) {
      theLayerHitsPerChamber[chamberId.rawId()].push_back(*layerItr);
    }
  }

}

namespace CSCSegmentValidationUtils {
  bool SimHitPabsLessThan(const PSimHit & p1, const PSimHit & p2)
  {
    return p1.pabs() < p2.pabs();
  }
}


const PSimHit * CSCSegmentValidation::keyHit(int chamberId) const
{
  const PSimHit * result = 0;
  int layerId = chamberId + 3;
  const edm::PSimHitContainer & layerHits = theSimHitMap->hits(layerId);

  if(!layerHits.empty())
  {
    // pick the hit with maximum energy
    edm::PSimHitContainer::const_iterator hitItr = std::max_element(layerHits.begin(), layerHits.end(),
                                                    CSCSegmentValidationUtils::SimHitPabsLessThan);
    result = &(*hitItr);
  }
  return result;
}   

