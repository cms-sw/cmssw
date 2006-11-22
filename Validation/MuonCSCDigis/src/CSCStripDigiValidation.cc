#include "Validation/MuonCSCDigis/src/CSCStripDigiValidation.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"


CSCStripDigiValidation::CSCStripDigiValidation(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe)
: dbe_(dbe),
  theInputTag(ps.getParameter<edm::InputTag>("stripDigiTag")),
  thePedestalSum(0),
  thePedestalCovarianceSum(0),
  thePedestalCount(0),
  thePedestalPlot( dbe_->book1D("CSCPedestal", "CSC Pedestal ", 400, 400, 800) ),
  thePedestalTimeCorrelationPlot(0),
  thePedestalNeighborCorrelationPlot(0),
  theAmplitudePlot( dbe_->book1D("CSCStripAmplitude", "CSC Strip Amplitude", 200, 0, 2000) ),
  theRatio4to5Plot( dbe_->book1D("CSCStrip4to5", "CSC Strip Ratio tbin 4 to tbin 5", 100, 0, 1) ),
  theRatio6to5Plot( dbe_->book1D("CSCStrip6to5", "CSC Strip Ratio tbin 6 to tbin 5", 120, 0, 1.2) ),
  theNDigisPerLayerPlot( dbe_->book1D("CSCStripDigisPerLayer", "Number of CSC Strip Digis per layer", 48, 0, 48) ),
  theNDigisPerChamberPlot(0),
  theNDigisPerEventPlot( dbe_->book1D("CSCStripDigisPerEvent", "Number of CSC Strip Digis per event", 100, 0, 500) )
{
}


CSCStripDigiValidation::~CSCStripDigiValidation() {}


void CSCStripDigiValidation::analyze(const edm::Event& e, const edm::EventSetup&)
{
  edm::Handle<CSCStripDigiCollection> strips;

  try {
    e.getByLabel(theInputTag, strips);
  } catch (...) {
    edm::LogError("CSCDigiDump") << "Cannot get strips by label " << theInputTag.encode();
  }

 unsigned nDigisPerEvent = 0;

 for (CSCStripDigiCollection::DigiRangeIterator j=strips->begin(); j!=strips->end(); j++) {
    std::vector<CSCStripDigi>::const_iterator digiItr = (*j).second.first;
    std::vector<CSCStripDigi>::const_iterator last = (*j).second.second;
    theNDigisPerLayerPlot->Fill(last-digiItr);

    for( ; digiItr != last; ++digiItr) {
      ++nDigisPerEvent;
      // average up the pedestals
      std::vector<int> adcCounts = digiItr->getADCCounts();
      thePedestalSum += adcCounts[0];
      thePedestalSum += adcCounts[1];
      thePedestalCount += 2;

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



