#include "Validation/MuonCSCDigis/src/CSCALCTDigiValidation.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/CSCGeometry/interface/CSCLayerGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "DQMServices/Core/interface/DQMStore.h"

CSCALCTDigiValidation::CSCALCTDigiValidation(const edm::InputTag & inputTag,
                                             edm::ConsumesCollector && iC):
  CSCBaseValidation(inputTag),
  theTimeBinPlots(),
  theNDigisPerLayerPlots()
{
  alcts_Token_ = iC.consumes<CSCALCTDigiCollection> (inputTag);
}

CSCALCTDigiValidation::~CSCALCTDigiValidation()
{
}

void CSCALCTDigiValidation::bookHistograms(DQMStore::IBooker & iBooker)
{
  theNDigisPerEventPlot = iBooker.book1D("CSCALCTDigisPerEvent", "CSC ALCT Digis per event", 100, 0, 100);
  for(int i = 0; i < 10; ++i)
  {
    char title1[200], title2[200];
    sprintf(title1, "CSCALCTDigiTimeType%d", i+1);
    sprintf(title2, "CSCALCTDigisPerLayerType%d", i+1);
    theTimeBinPlots[i] = iBooker.book1D(title1, title1, 20, 0, 20);
    theNDigisPerLayerPlots[i] = iBooker.book1D(title2, title2, 100, 0, 20);
  }
}

void CSCALCTDigiValidation::analyze(const edm::Event&e, const edm::EventSetup&)
{
  edm::Handle<CSCALCTDigiCollection> alcts;

  e.getByToken(alcts_Token_, alcts);
  if (!alcts.isValid()) {
    edm::LogError("CSCDigiDump") << "Cannot get alcts by label " << theInputTag.encode();
  }
  unsigned nDigisPerEvent = 0;

  for (CSCALCTDigiCollection::DigiRangeIterator j=alcts->begin(); j!=alcts->end(); j++) {
    std::vector<CSCALCTDigi>::const_iterator beginDigi = (*j).second.first;
    std::vector<CSCALCTDigi>::const_iterator endDigi = (*j).second.second;
    CSCDetId detId((*j).first.rawId());
    int chamberType = detId.iChamberType();
    int nDigis = endDigi-beginDigi;
    nDigisPerEvent += nDigis;
    theNDigisPerLayerPlots[chamberType-1]->Fill(nDigis);

    for( std::vector<CSCALCTDigi>::const_iterator digiItr = beginDigi;
         digiItr != endDigi; ++digiItr)
    {
      theTimeBinPlots[chamberType-1]->Fill(digiItr->getBX());
    }
  }
}
