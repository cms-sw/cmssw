#include "Validation/MuonCSCDigis/src/CSCComparatorDigiValidation.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/DQMStore.h"


CSCComparatorDigiValidation::CSCComparatorDigiValidation(
    const edm::InputTag & inputTag,
    const edm::InputTag & stripDigiInputTag,
    edm::ConsumesCollector && iC):
  CSCBaseValidation(inputTag),
  theStripDigi_Token_(iC.consumes<CSCStripDigiCollection>(stripDigiInputTag)),
  theTimeBinPlots(),
  theNDigisPerLayerPlots(),
  theStripDigiPlots(),
  the3StripPlots()
{
  comparators_Token_ = iC.consumes<CSCComparatorDigiCollection>(inputTag);
}

CSCComparatorDigiValidation::~CSCComparatorDigiValidation()
{
}

void CSCComparatorDigiValidation::bookHistograms(DQMStore::IBooker & iBooker)
{
  theNDigisPerEventPlot = iBooker.book1D("CSCComparatorDigisPerEvent", "CSC Comparator Digis per event", 100, 0, 100);
  for(int i = 0; i < 10; ++i)
  {
    char title1[200], title2[200], title3[200], title4[200];
    sprintf(title1, "CSCComparatorDigiTimeType%d", i+1);
    sprintf(title2, "CSCComparatorDigisPerLayerType%d", i+1);
    sprintf(title3, "CSCComparatorStripAmplitudeType%d", i+1);
    sprintf(title4, "CSCComparator3StripAmplitudeType%d", i+1);
    theTimeBinPlots[i] = iBooker.book1D(title1, title1, 9, 0, 8);
    theNDigisPerLayerPlots[i] = iBooker.book1D(title2, title2, 100, 0, 20);
    theStripDigiPlots[i] = iBooker.book1D(title3, title3, 100, 0, 1000);
    the3StripPlots[i] = iBooker.book1D(title4, title4, 100, 0, 1000);
  }
}

void CSCComparatorDigiValidation::analyze(const edm::Event&e, const edm::EventSetup&)
{
  edm::Handle<CSCComparatorDigiCollection> comparators;
  edm::Handle<CSCStripDigiCollection> stripDigis;

  e.getByToken(comparators_Token_, comparators);
  if (!comparators.isValid()) {
    edm::LogError("CSCDigiDump") << "Cannot get comparators by label " << theInputTag.encode();
  }
  e.getByToken(theStripDigi_Token_, stripDigis);
  if (!stripDigis.isValid()) {
    edm::LogError("CSCDigiDump") << "Cannot get comparators by label " << theInputTag.encode();
  }

  unsigned nDigisPerEvent = 0;

  for (CSCComparatorDigiCollection::DigiRangeIterator j=comparators->begin(); j!=comparators->end(); j++) {
    std::vector<CSCComparatorDigi>::const_iterator digiItr = (*j).second.first;
    std::vector<CSCComparatorDigi>::const_iterator last = (*j).second.second;

    CSCDetId detId((*j).first);
    const CSCLayer * layer = findLayer(detId.rawId());
    int chamberType = layer->chamber()->specs()->chamberType();

    CSCStripDigiCollection::Range stripRange = stripDigis->get(detId);

    theNDigisPerLayerPlots[chamberType-1]->Fill(last-digiItr);

    for( ; digiItr != last; ++digiItr) {
      ++nDigisPerEvent;
      theTimeBinPlots[chamberType-1]->Fill(digiItr->getTimeBin());

      int strip = digiItr->getStrip();
      for(std::vector<CSCStripDigi>::const_iterator stripItr = stripRange.first;
          stripItr != stripRange.second; ++stripItr)
      {
        if(stripItr->getStrip() == strip)
        {
          std::vector<int> adc = stripItr->getADCCounts();
          float pedc = 0.5*(adc[0]+adc[1]);
          float amp = adc[4] - pedc;
          theStripDigiPlots[chamberType-1]->Fill(amp);
          // check neighbors
          if(stripItr != stripRange.first && stripItr != stripRange.second-1)
          {
            std::vector<int> adcl = (stripItr-1)->getADCCounts();
            std::vector<int> adcr = (stripItr+1)->getADCCounts();
            float pedl = 0.5*(adcl[0]+adcl[1]);
            float pedr = 0.5*(adcr[0]+adcr[1]);
            float three = adcl[4]-pedl
                        + adcr[4]-pedr
                        + amp;
            the3StripPlots[chamberType-1]->Fill(three);
          }
        }
      }
    }

  }

  theNDigisPerEventPlot->Fill(nDigisPerEvent);
}

