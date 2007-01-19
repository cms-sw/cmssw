#include "Validation/MuonCSCDigis/src/CSCWireDigiValidation.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"

#include "Geometry/CSCGeometry/interface/CSCLayerGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"



CSCWireDigiValidation::CSCWireDigiValidation(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe,
                                             const PSimHitMap & hitMap)
: dbe_(dbe),
  theInputTag(ps.getParameter<edm::InputTag>("wireDigiTag")),
  theSimHitMap(hitMap),
  theTimeBinPlots(),
  theNDigisPerLayerPlots(),
  theNDigisPerEventPlot( dbe_->book1D("CSCWireDigisPerEvent", "CSC Wire Digis per event", 100, 0, 100) )
{
  for(int i = 0; i < 10; ++i)
  {
    char title1[200], title2[200], title3[200];
    sprintf(title1, "CSCWireDigiTimeType%d", i+1);
    sprintf(title2, "CSCWireDigisPerLayerType%d", i+1);
    sprintf(title3, "CSCWireDigiResolution%d", i+1);
    theTimeBinPlots[i] = dbe_->book1D(title1, title1, 9, 0, 8);
    theNDigisPerLayerPlots[i] = dbe_->book1D(title2, title2, 100, 0, 20);
    theResolutionPlots[i] = dbe_->book1D(title3, title3, 100, -10, 10);
  }
}



CSCWireDigiValidation::~CSCWireDigiValidation()
{
}


void CSCWireDigiValidation::analyze(const edm::Event&e, const edm::EventSetup&)
{
  edm::Handle<CSCWireDigiCollection> wires;

  try {
    e.getByLabel(theInputTag, wires);
  } catch (...) {
    edm::LogError("CSCDigiDump") << "Cannot get wires by label " << theInputTag.encode();
  }

  unsigned nDigisPerEvent = 0;

  for (CSCWireDigiCollection::DigiRangeIterator j=wires->begin(); j!=wires->end(); j++) {
    std::vector<CSCWireDigi>::const_iterator beginDigi = (*j).second.first;
    std::vector<CSCWireDigi>::const_iterator endDigi = (*j).second.second;
    int detId = (*j).first.rawId();
    
    const CSCLayer * layer = findLayer(detId);
    int chamberType = layer->chamber()->specs()->chamberType();
    int nDigis = endDigi-beginDigi;
    nDigisPerEvent += nDigis;
    theNDigisPerLayerPlots[chamberType-1]->Fill(nDigis);

    for( std::vector<CSCWireDigi>::const_iterator digiItr = beginDigi;
         digiItr != endDigi; ++digiItr) 
    {
      theTimeBinPlots[chamberType-1]->Fill(digiItr->getTimeBin());
    }

    const edm::PSimHitContainer simHits = theSimHitMap.hits(detId);
    if(nDigis == 1 && simHits.size() == 1)
    {
      plotResolution(simHits[0], *beginDigi, layer, chamberType);
    }
  }

  theNDigisPerEventPlot->Fill(nDigisPerEvent);
}


void CSCWireDigiValidation::plotResolution(const PSimHit & hit, const CSCWireDigi & digi, 
                                           const CSCLayer * layer, int chamberType)
{
  double hitX = hit.localPosition().x();
  double hitY = hit.localPosition().y();
  double digiY = layer->geometry()->yOfWireGroup(digi.getWireGroup(), hitX);
  theResolutionPlots[chamberType-1]->Fill(digiY - hitY);
}


const CSCLayer * CSCWireDigiValidation::findLayer(int detId) const {
  assert(theCSCGeometry != 0);
  const GeomDetUnit* detUnit = theCSCGeometry->idToDetUnit(CSCDetId(detId));
  return dynamic_cast<const CSCLayer *>(detUnit);
}

