#include "SimMuon/CSCDigitizer/src/CSCDigiSuppressor.h"
#include "DataFormats/Common/interface/Handle.h"
#include <DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h>
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"


CSCDigiSuppressor::CSCDigiSuppressor(const edm::ParameterSet& ps)
: theLCTTag(ps.getParameter<edm::InputTag>("lctTag")),
  theStripDigiTag(ps.getParameter<edm::InputTag>("stripDigiTag")),
  theStripElectronicsSim(ps),
  theStripConditions(new CSCDbStripConditions(ps))
{
std::cout << "MAKE?" << std::endl;
  produces<CSCStripDigiCollection>("MuonCSCSuppressedStripDigi");

  edm::Service<edm::RandomNumberGenerator> rng;
  if ( ! rng.isAvailable()) {
   throw cms::Exception("Configuration")
     << "CSCDigitizer requires the RandomNumberGeneratorService\n"
        "which is not present in the configuration file.  You must add the service\n"
        "in the configuration file or remove the modules that require it.";
  }

  CLHEP::HepRandomEngine& engine = rng->getEngine();

  theStripElectronicsSim.setRandomEngine(engine);
  theStripConditions->setRandomEngine(engine);
  theStripElectronicsSim.setStripConditions(theStripConditions);
std::cout << "MADE" << std::endl;
}


void CSCDigiSuppressor::produce(edm::Event& e, const edm::EventSetup& eventSetup) 
{
  edm::Handle<CSCStripDigiCollection> oldStripDigis;
  e.getByLabel(theStripDigiTag, oldStripDigis);
  if (!oldStripDigis.isValid()) {
    edm::LogError("CSCDigiValidation") << "Cannot get strips by label "
                                       << theStripDigiTag.encode();
  }

  edm::Handle<CSCCorrelatedLCTDigiCollection> lcts;
  e.getByLabel(theLCTTag, lcts);
std::cout << "GOTLCT" << std::endl;
  std::auto_ptr<CSCStripDigiCollection> newStripDigis(new CSCStripDigiCollection());

  theStripConditions->initializeEvent(eventSetup);


  for(CSCCorrelatedLCTDigiCollection::DigiRangeIterator detUnitItr = lcts->begin();
      detUnitItr != lcts->end(); ++detUnitItr)
  {
    const CSCDetId& id = (*detUnitItr).first;
    const CSCCorrelatedLCTDigiCollection::Range& range = (*detUnitItr).second;
    std::list<int> keyStrips;
    for (CSCCorrelatedLCTDigiCollection::const_iterator digiItr = range.first;
         digiItr != range.second; digiItr++) 
    {
      // convert from 0..159 to 1..80
      keyStrips.push_back(digiItr->getStrip()/2+1);
    }

    fillDigis(id, keyStrips, *oldStripDigis, *newStripDigis);
  }

  e.put(newStripDigis, "MuonCSCSuppressedStripDigi");
}


void CSCDigiSuppressor::fillDigis(const CSCDetId & id, const std::list<int> & keyStrips, 
                                       const CSCStripDigiCollection & oldStripDigis,
                                       CSCStripDigiCollection & newStripDigis)
{
std::cout << "FILLDIGIS" << std::endl;
  std::list<int> cfebs = cfebsToRead(id, keyStrips);
  std::list<int> strips = stripsToRead(cfebs);
  CSCStripDigiCollection::Range chamberDigis = oldStripDigis.get(id);
  // strips are sorted by layer
  for(int layer = 1; layer <= 6; ++layer)
  {
std::cout << "LAYER" << layer << std::endl;
    // make a copy so we can mangle it
    std::list<int> layerStrips = strips;
    CSCDetId layerId(id.rawId()+layer);
    theStripElectronicsSim.setLayerId(layerId);

    for(CSCStripDigiCollection::const_iterator digiItr = chamberDigis.first;
        digiItr != chamberDigis.second; ++digiItr)
    {
      std::list<int>::iterator stripsToDoItr = std::find(layerStrips.begin(), layerStrips.end(), digiItr->getStrip());
      // if it's found, save the digi and check the strip off the list
      if(stripsToDoItr != strips.end())
      {
        newStripDigis.insertDigi(layerId, *digiItr);
        layerStrips.erase(stripsToDoItr);
      }
    }

    // whatever is left over needs to have its own noise generated
    for(std::list<int>::iterator leftoverStripItr = layerStrips.begin();
        leftoverStripItr != layerStrips.end(); ++leftoverStripItr)
    {
std::cout << "LEFTOVER " << *leftoverStripItr << std::endl;
      CSCAnalogSignal noiseSignal = theStripElectronicsSim.makeNoiseSignal(*leftoverStripItr);
    }
  }
}


std::list<int>
CSCDigiSuppressor::cfebsToRead(const CSCDetId & id, const std::list<int> & keyStrips) const
{
  // always accept ME1A, because it's too much trouble looking
  // for LCTs in ME11
  if(id.station() == 1 && id.ring() == 4)
  {
    return std::list<int>(1, 0);
  }

  int maxCFEBs = (id.station() == 1) ? 4 : 5;
  if(id.station() == 1 && id.ring() == 2) maxCFEBs = 5;

  //copied from CSCStripElectronicsSim
  std::list<int> cfebs;
  for(std::list<int>::const_iterator keyStripItr = keyStrips.begin(); 
      keyStripItr != keyStrips.end(); ++keyStripItr)
  {
    int cfeb = ((*keyStripItr)-1)/16;
    cfebs.push_back(cfeb);
    int remainder = ((*keyStripItr)-1)%16;
    // if we're within 3 strips of an edge, take neighboring CFEB, too
    if(remainder <= 2 && cfeb != 0)
    {
      cfebs.push_back(cfeb-1);
    }

    if(remainder >= 13 && cfeb < maxCFEBs) 
    {
      cfebs.push_back(cfeb+1);
    }
  }
  cfebs.sort();
  cfebs.unique();
  return cfebs;
}


std::list<int>
CSCDigiSuppressor::stripsToRead(const std::list<int> & cfebs) const
{
  std::list<int> strips;
  for(std::list<int>::const_iterator cfebItr = cfebs.begin();
      cfebItr != cfebs.end(); ++cfebItr)
  {
    for(int i = 1; i <= 16; ++i)
    {
      strips.push_back((*cfebItr)*16+i);
    }
  }
  return strips;
}


