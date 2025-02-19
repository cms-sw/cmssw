#include "Validation/MuonCSCDigis/src/CSCStripDigiValidation.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DQMServices/Core/interface/DQMStore.h"


CSCStripDigiValidation::CSCStripDigiValidation(DQMStore* dbe, 
                                               const edm::InputTag & inputTag,
                                               bool doSim)
: CSCBaseValidation(dbe, inputTag),
  thePedestalSum(0),
  thePedestalCovarianceSum(0),
  thePedestalCount(0),
  theDoSimFlag(doSim),
  thePedestalPlot( dbe_->book1D("CSCPedestal", "CSC Pedestal ", 400, 550, 650) ),
  thePedestalTimeCorrelationPlot(0),
  thePedestalNeighborCorrelationPlot(0),
  theAmplitudePlot( dbe_->book1D("CSCStripAmplitude", "CSC Strip Amplitude", 200, 0, 2000) ),
  theRatio4to5Plot( dbe_->book1D("CSCStrip4to5", "CSC Strip Ratio tbin 4 to tbin 5", 100, 0, 1) ),
  theRatio6to5Plot( dbe_->book1D("CSCStrip6to5", "CSC Strip Ratio tbin 6 to tbin 5", 120, 0, 1.2) ),
  theNDigisPerLayerPlot( dbe_->book1D("CSCStripDigisPerLayer", "Number of CSC Strip Digis per layer", 48, 0, 48) ),
  theNDigisPerChamberPlot(0),
  theNDigisPerEventPlot( dbe_->book1D("CSCStripDigisPerEvent", "Number of CSC Strip Digis per event", 100, 0, 500) )
{
  if(doSim) {
    for(int i = 0; i < 10; ++i)
    {
      char title1[200];
      sprintf(title1, "CSCStripDigiResolution%d", i+1);
      theResolutionPlots[i] = dbe_->book1D(title1, title1, 100, -5, 5);
    }
  }

}


CSCStripDigiValidation::~CSCStripDigiValidation() {
 
//   edm::LogInfo("CSCDigiValidation") << "RATIO for strips 4 to 5 : " << theRatio4to5Plot->getMean();
//   edm::LogInfo("CSCDigiValidation") << "RATIO for strips 6 to 5 : " << theRatio6to5Plot->getMean();
//   edm::LogInfo("CSCDigiValidation") << "NDIGIS per event : " << theNDigisPerEventPlot->getMean();

}


void CSCStripDigiValidation::analyze(const edm::Event& e, 
                                     const edm::EventSetup&)
{
  edm::Handle<CSCStripDigiCollection> strips;
  e.getByLabel(theInputTag, strips);
  if (!strips.isValid()) {
    edm::LogError("CSCDigiValidation") << "Cannot get strips by label " 
                                       << theInputTag.encode();
  }

 unsigned nDigisPerEvent = 0;

 for (CSCStripDigiCollection::DigiRangeIterator j=strips->begin(); j!=strips->end(); j++) {
    std::vector<CSCStripDigi>::const_iterator digiItr = (*j).second.first;
    std::vector<CSCStripDigi>::const_iterator last = (*j).second.second;
    int nDigis = last-digiItr;
    nDigisPerEvent += nDigis;
    theNDigisPerLayerPlot->Fill(nDigis);

    double maxAmplitude = 0.;
    // int maxStrip = 0;

    for( ; digiItr != last; ++digiItr) {
      // average up the pedestals
      std::vector<int> adcCounts = digiItr->getADCCounts();
      thePedestalSum += adcCounts[0];
      thePedestalSum += adcCounts[1];
      thePedestalCount += 2;
      float pedestal = thePedestalSum/thePedestalCount;
      if(adcCounts[4]-pedestal > maxAmplitude)
      {
      //  maxStrip = digiItr->getStrip();
        maxAmplitude = adcCounts[4]-pedestal;
      }

      // if we have enough pedestal statistics
      if(thePedestalCount > 100)
      {
        fillPedestalPlots(*digiItr);
       
        // see if it's big enough to count as "signal"
        if(adcCounts[5] > (thePedestalSum/thePedestalCount + 100))
        {
          fillSignalPlots(*digiItr);
        }
      }
    }

/*
    int detId = (*j).first.rawId();

    edm::PSimHitContainer simHits = theSimHitMap->hits(detId);

    if(simHits.size() == 1)
    {
      const CSCLayer * layer = findLayer(detId);
      int chamberType = layer->chamber()->specs()->chamberType();
      plotResolution(simHits[0], maxStrip, layer, chamberType);
    }
*/
  } // loop over digis

  theNDigisPerEventPlot->Fill(nDigisPerEvent);
}


void CSCStripDigiValidation::fillPedestalPlots(const CSCStripDigi & digi)
{
  std::vector<int> adcCounts = digi.getADCCounts();
  thePedestalPlot->Fill(adcCounts[0]);
  thePedestalPlot->Fill(adcCounts[1]);
}



void CSCStripDigiValidation::fillSignalPlots(const CSCStripDigi & digi)
{
  std::vector<int> adcCounts = digi.getADCCounts();
  float pedestal = thePedestalSum/thePedestalCount;
  theAmplitudePlot->Fill(adcCounts[4] - pedestal);
  theRatio4to5Plot->Fill( (adcCounts[3]-pedestal) / (adcCounts[4]-pedestal) );
  theRatio6to5Plot->Fill( (adcCounts[5]-pedestal) / (adcCounts[4]-pedestal) );
}


void CSCStripDigiValidation::plotResolution(const PSimHit & hit, int strip,
                                           const CSCLayer * layer, int chamberType)
{
  double hitX = hit.localPosition().x();
  double hitY = hit.localPosition().y();
  double digiX = layer->geometry()->xOfStrip(strip, hitY);
  theResolutionPlots[chamberType-1]->Fill(digiX - hitX);
}


