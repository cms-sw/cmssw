#include "Validation/MuonCSCDigis/src/CSCCLCTDigiValidation.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h"

#include "Geometry/CSCGeometry/interface/CSCLayerGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "DQMServices/Core/interface/DQMStore.h"



CSCCLCTDigiValidation::CSCCLCTDigiValidation(DQMStore* dbe, const edm::InputTag & inputTag)
: CSCBaseValidation(dbe, inputTag),
  theTimeBinPlots(),
  theNDigisPerLayerPlots(),
  theNDigisPerEventPlot( dbe_->book1D("CSCCLCTDigisPerEvent", "CSC CLCT Digis per event", 100, 0, 100) )
{
  for(int i = 0; i < 10; ++i)
  {
    char title1[200], title2[200];
    sprintf(title1, "CSCCLCTDigiTimeType%d", i+1);
    sprintf(title2, "CSCCLCTDigisPerLayerType%d", i+1);
    theTimeBinPlots[i] = dbe_->book1D(title1, title1, 20, 0, 20);
    theNDigisPerLayerPlots[i] = dbe_->book1D(title2, title2, 100, 0, 20);
  }
}



CSCCLCTDigiValidation::~CSCCLCTDigiValidation()
{

//   for(int i = 0; i < 10; ++i)
//   {
//     edm::LogInfo("CSCDigiValidation") << "Mean of " << theTimeBinPlots[i]->getName() 
//       << " is " << theTimeBinPlots[i]->getMean() 
//       << " +/- " << theTimeBinPlots[i]->getRMS();
//   }

}


void CSCCLCTDigiValidation::analyze(const edm::Event&e, const edm::EventSetup&)
{
  edm::Handle<CSCCLCTDigiCollection> clcts;

  e.getByLabel(theInputTag, clcts);
  if (!clcts.isValid()) {
    edm::LogError("CSCDigiDump") << "Cannot get clcts by label " << theInputTag.encode();
  }

  unsigned nDigisPerEvent = 0;

  for (CSCCLCTDigiCollection::DigiRangeIterator j=clcts->begin(); j!=clcts->end(); j++) {
    std::vector<CSCCLCTDigi>::const_iterator beginDigi = (*j).second.first;
    std::vector<CSCCLCTDigi>::const_iterator endDigi = (*j).second.second;
    CSCDetId detId((*j).first.rawId());
    int chamberType = detId.iChamberType();
 
    int nDigis = endDigi-beginDigi;
    nDigisPerEvent += nDigis;
    theNDigisPerLayerPlots[chamberType-1]->Fill(nDigis);

    for( std::vector<CSCCLCTDigi>::const_iterator digiItr = beginDigi;
         digiItr != endDigi; ++digiItr) 
    {
      theTimeBinPlots[chamberType-1]->Fill(digiItr->getBX());
    }

  }
}
