#include "Validation/MuonCSCDigis/src/CSCComparatorDigiValidation.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"


CSCComparatorDigiValidation::CSCComparatorDigiValidation(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe,
                                                         const PSimHitMap & hitMap)
: dbe_(dbe),
  theInputTag(ps.getParameter<edm::InputTag>("comparatorDigiTag")),
  theSimHitMap(hitMap),
  theTimeBinPlots(),
  theNDigisPerLayerPlots(),
  theNDigisPerEventPlot( dbe_->book1D("CSCComparatorDigisPerEvent", "CSC Comparator Digis per event", 100, 0, 100) )
{
  for(int i = 0; i < 10; ++i)
  {
    char title1[200], title2[200];
    sprintf(title1, "CSCComparatorDigiTimeType%d", i+1);
    sprintf(title2, "CSCComparatorDigisPerLayerType%d", i+1);
    theTimeBinPlots[i] = dbe_->book1D(title1, title1, 9, 0, 8);
    theNDigisPerLayerPlots[i] = dbe_->book1D(title2, title2, 100, 0, 20);
  }
}



CSCComparatorDigiValidation::~CSCComparatorDigiValidation()
{
}


void CSCComparatorDigiValidation::analyze(const edm::Event&e, const edm::EventSetup&)
{
  edm::Handle<CSCComparatorDigiCollection> comparators;

  try {
    e.getByLabel(theInputTag, comparators);
  } catch (...) {
    edm::LogError("CSCDigiDump") << "Cannot get comparators by label " << theInputTag.encode();
  }

  unsigned nDigisPerEvent = 0;

  for (CSCComparatorDigiCollection::DigiRangeIterator j=comparators->begin(); j!=comparators->end(); j++) {
    std::vector<CSCComparatorDigi>::const_iterator digiItr = (*j).second.first;
    std::vector<CSCComparatorDigi>::const_iterator last = (*j).second.second;

    CSCDetId detId((*j).first);
    // TODO
    int chamberType = 1;

    theNDigisPerLayerPlots[chamberType-1]->Fill(last-digiItr);

    for( ; digiItr != last; ++digiItr) {
      ++nDigisPerEvent;
      theTimeBinPlots[chamberType-1]->Fill(digiItr->getTimeBin());
    }

  }

  theNDigisPerEventPlot->Fill(nDigisPerEvent);
}

