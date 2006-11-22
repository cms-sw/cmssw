#include "Validation/MuonCSCDigis/src/CSCWireDigiValidation.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"


CSCWireDigiValidation::CSCWireDigiValidation(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe)
: dbe_(dbe),
  theInputTag(ps.getParameter<edm::InputTag>("wireDigiTag")),
  theTimeBinPlots(),
  theNDigisPerLayerPlots(),
  theNDigisPerEventPlot( dbe_->book1D("CSCWireDigisPerEvent", "CSC Wire Digis per event", 100, 0, 100) )
{
  for(int i = 0; i < 10; ++i)
  {
    char title1[200], title2[200];
    sprintf(title1, "CSCWireDigiTimeType%d", i+1);
    sprintf(title2, "CSCWireDigisPerLayerType%d", i+1);
    theTimeBinPlots[i] = dbe_->book1D(title1, title1, 9, 0, 8);
    theNDigisPerLayerPlots[i] = dbe_->book1D(title2, title2, 100, 0, 20);
  }
}



CSCWireDigiValidation::~CSCWireDigiValidation()
{
}


void CSCWireDigiValidation::analyze(const edm::Event&e, const edm::EventSetup&)
{
std::cout << "CSCWireDigiValidation" << std::endl;
  edm::Handle<CSCWireDigiCollection> wires;

  try {
    e.getByLabel(theInputTag, wires);
  } catch (...) {
    edm::LogError("CSCDigiDump") << "Cannot get wires by label " << theInputTag.encode();
  }

  unsigned nDigisPerEvent = 0;

  for (CSCWireDigiCollection::DigiRangeIterator j=wires->begin(); j!=wires->end(); j++) {
    std::vector<CSCWireDigi>::const_iterator digiItr = (*j).second.first;
    std::vector<CSCWireDigi>::const_iterator last = (*j).second.second;

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

