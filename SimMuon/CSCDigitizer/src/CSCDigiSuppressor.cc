#include "SimMuon/CSCDigitizer/src/CSCDigiSuppressor.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include <algorithm>


CSCDigiSuppressor::CSCDigiSuppressor(const edm::ParameterSet& ps)
: theLCTLabel(ps.getParameter<std::string>("lctLabel")),
  theDigiLabel(ps.getParameter<std::string>("digiLabel")),
  theStripElectronicsSim(ps),
  theStripConditions(new CSCDbStripConditions(ps))
{
  produces<CSCStripDigiCollection>("MuonCSCStripDigi");

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
}


void CSCDigiSuppressor::produce(edm::Event& e, const edm::EventSetup& eventSetup) 
{
  edm::Handle<CSCStripDigiCollection> oldStripDigis;
  e.getByLabel(theDigiLabel, "MuonCSCStripDigi", oldStripDigis);
  if (!oldStripDigis.isValid()) {
    edm::LogError("CSCDigiValidation") << "Cannot get strips by label "
                                       << theDigiLabel;
  }

  edm::Handle<CSCCorrelatedLCTDigiCollection> lcts;
  e.getByLabel(theLCTLabel, lcts);

  std::auto_ptr<CSCStripDigiCollection> newStripDigis(new CSCStripDigiCollection());

  theStripConditions->initializeEvent(eventSetup);

  for(CSCCorrelatedLCTDigiCollection::DigiRangeIterator detUnitItr = lcts->begin();
      detUnitItr != lcts->end(); ++detUnitItr)
  {
    const CSCDetId& id = (*detUnitItr).first;
    std::cout << "LCT IN " << id <<std::endl;
    const CSCCorrelatedLCTDigiCollection::Range& range = (*detUnitItr).second;
    std::list<int> keyStrips;
    for (CSCCorrelatedLCTDigiCollection::const_iterator digiItr = range.first;
         digiItr != range.second; digiItr++) 
    {
      // convert from 0..159 to 1..80
      keyStrips.push_back(digiItr->getStrip()/2+1);
    }

    fillDigis(id, keyStrips, *oldStripDigis, *newStripDigis);
    // Don;t suppress real signal in ME1/A, but don't creat noise, either
    if(id.station() == 1 && id.ring() == 1)
    {
      CSCDetId me1aId(id.endcap(), id.station(), 4, id.chamber(), 0);
      fillDigis(me1aId, keyStrips, *oldStripDigis, *newStripDigis);
    }
  }

  e.put(newStripDigis, "MuonCSCStripDigi");
  suppressWires(e);
}


void CSCDigiSuppressor::suppressWires(edm::Event & e)
{
  edm::Handle<CSCWireDigiCollection> oldWireDigis;
  e.getByLabel(theDigiLabel, "MuonCSCWireDigi", oldWireDigis);
  std::auto_ptr<CSCWireDigiCollection> newWireDigis(new CSCWireDigiCollection(*oldWireDigis));

  edm::Handle<CSCALCTDigiCollection> alctDigis;
  e.getByLabel(theLCTLabel, alctDigis);
  typedef CSCALCTDigiCollection::DigiRangeIterator ALCTItr;
  typedef CSCWireDigiCollection::DigiRangeIterator WireItr;

  for(WireItr wireItr = oldWireDigis->begin();
      wireItr != oldWireDigis->end(); ++wireItr)
  {
    const CSCDetId& id = (*wireItr).first;
    bool found = false;
    for(ALCTItr alctItr = alctDigis->begin(); 
        !found && alctItr != alctDigis->end(); ++alctItr)
    {
      if((*alctItr).first == id) 
      {
        found = true;
      }
    }
    if(found)
    {
      newWireDigis->put((*wireItr).second, (*wireItr).first);
    }
  }

  e.put(newWireDigis, "MuonCSCWireDigi");
}


void CSCDigiSuppressor::fillDigis(const CSCDetId & id, const std::list<int> & keyStrips, 
                                       const CSCStripDigiCollection & oldStripDigis,
                                       CSCStripDigiCollection & newStripDigis)
{
  std::list<int> cfebs = cfebsToRead(id, keyStrips);
  std::cout << "CFEBS TO READ FOR  " << id << ": ";
  for(std::list<int>::const_iterator i = cfebs.begin(); i != cfebs.end(); ++i){
    std::cout << *i << " " ;
  }
std::cout << std::endl;
  std::list<int> strips = stripsToRead(cfebs);
  // strips are sorted by layer
  for(int layer = 1; layer <= 6; ++layer)
  {
    // make a copy so we can mangle it
    std::list<int> layerStrips = strips;
    CSCDetId layerId(id.rawId()+layer);
    std::vector<CSCStripDigi> newDigis;
    theStripElectronicsSim.setLayerId(layerId);
    CSCStripDigiCollection::Range layerDigis = oldStripDigis.get(layerId);
std::cout << "STRIPDIGI " << layerId << " " << layerDigis.second-layerDigis.first << std::endl;
    for (std::vector<CSCStripDigi>::const_iterator digiItr=layerDigis.first; 
         digiItr!=layerDigis.second; digiItr++) 
    {
      std::list<int>::iterator stripsToDoItr = std::find(layerStrips.begin(), layerStrips.end(), digiItr->getStrip());
if(layerId.ring() == 4) {
  std::cout << "RING4 " << " FOUND " << (stripsToDoItr != layerStrips.end()) << std::endl;
}

      // if it's found, save the digi and check the strip off the list
      if(stripsToDoItr != layerStrips.end())
      {
        newDigis.push_back(*digiItr);
        layerStrips.erase(stripsToDoItr);
      }
      else  
      {
        //std::cout << "SUPPRESSION IN " << layerId << " " << digiItr->getStrip() << std::endl;
      }
    }

    // whatever is left over needs to have its own noise generated
    // don't generate noise for ME1/A
    if(layerId.ring() != 4) 
    {
      for(std::list<int>::iterator leftoverStripItr = layerStrips.begin();
          leftoverStripItr != layerStrips.end(); ++leftoverStripItr)
      {
        CSCAnalogSignal noiseSignal = theStripElectronicsSim.makeNoiseSignal(*leftoverStripItr);
        theStripElectronicsSim.createDigi(layerId, noiseSignal, newDigis);
      }
    }

    if(!newDigis.empty())
    {
      // copy the digis into the collection
      CSCStripDigiCollection::Range digiRange(newDigis.begin(), newDigis.end()); 
      newStripDigis.put(digiRange, layerId);
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
