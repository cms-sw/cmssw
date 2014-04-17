#include "SimMuon/CSCDigitizer/src/CSCDigitizer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimMuon/CSCDigitizer/src/CSCDetectorHit.h"
#include "SimMuon/CSCDigitizer/src/CSCWireHitSim.h"
#include "SimMuon/CSCDigitizer/src/CSCStripHitSim.h"
#include "SimMuon/CSCDigitizer/src/CSCDriftSim.h"
#include "SimMuon/CSCDigitizer/src/CSCWireElectronicsSim.h"
#include "SimMuon/CSCDigitizer/src/CSCStripElectronicsSim.h"
#include "SimMuon/CSCDigitizer/src/CSCNeutronReader.h"
#include "SimMuon/CSCDigitizer/src/CSCStripConditions.h"
#include "Geometry/CSCGeometry/interface/CSCLayer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>


CSCDigitizer::CSCDigitizer(const edm::ParameterSet & p)
: theDriftSim(new CSCDriftSim()),
  theWireHitSim(new CSCWireHitSim(theDriftSim)),
  theStripHitSim(new CSCStripHitSim()),
  theWireElectronicsSim(new CSCWireElectronicsSim(p.getParameter<edm::ParameterSet>("wires"))),
  theStripElectronicsSim(new CSCStripElectronicsSim(p.getParameter<edm::ParameterSet>("strips"))),
  theNeutronReader(0),
  theCSCGeometry(0),
  theLayersNeeded(p.getParameter<unsigned int>("layersNeeded")),
  digitizeBadChambers_(p.getParameter<bool>("digitizeBadChambers"))
{
  if(p.getParameter<bool>("doNeutrons"))
  {
    theNeutronReader = new CSCNeutronReader(p.getParameter<edm::ParameterSet>("neutrons"));
  }
}


CSCDigitizer::~CSCDigitizer() {
  delete theNeutronReader;
  delete theStripElectronicsSim;
  delete theWireElectronicsSim;
  delete theStripHitSim;
  delete theWireHitSim;
  delete theDriftSim;
}



void CSCDigitizer::doAction(MixCollection<PSimHit> & simHits, 
                            CSCWireDigiCollection & wireDigis, 
                            CSCStripDigiCollection & stripDigis, 
                            CSCComparatorDigiCollection & comparators,
                            DigiSimLinks & wireDigiSimLinks,
                            DigiSimLinks & stripDigiSimLinks) 
{
  // arrange the hits by layer
  std::map<int, edm::PSimHitContainer> hitMap;
  for(MixCollection<PSimHit>::MixItr hitItr = simHits.begin();
      hitItr != simHits.end(); ++hitItr) 
  {
    hitMap[hitItr->detUnitId()].push_back(*hitItr);
  }

  // count how many layers on each chamber are hit
  std::map<int, std::set<int> > layersInChamberHit;
  for(std::map<int, edm::PSimHitContainer>::const_iterator hitMapItr = hitMap.begin();
      hitMapItr != hitMap.end(); ++hitMapItr)
  {
    CSCDetId cscDetId(hitMapItr->first); 
    int chamberId = cscDetId.chamberId();
    layersInChamberHit[chamberId].insert(cscDetId.layer());
  }

  // add neutron background, if needed
  if(theNeutronReader != 0)
  {
    theNeutronReader->addHits(hitMap);
  }

  // now loop over layers and run the simulation for each one
  for(std::map<int, edm::PSimHitContainer>::const_iterator hitMapItr = hitMap.begin();
      hitMapItr != hitMap.end(); ++hitMapItr)
  {
    CSCDetId detId = CSCDetId(hitMapItr->first);
    int chamberId = detId.chamberId();
    int endc = detId.endcap();
    int stat = detId.station();
    int ring = detId.ring();
    int cham = detId.chamber();
    
    unsigned int nLayersInChamberHitForWireDigis = 0;
    if (stat == 1 && ring == 1) { // ME1b
        std::set<int> layersInME1a = layersInChamberHit[CSCDetId(endc,stat,4,cham,0)];
        std::set<int> layersInME11 = layersInChamberHit[chamberId];
        layersInME11.insert(layersInME1a.begin(),layersInME1a.end());
        nLayersInChamberHitForWireDigis = layersInME11.size();
    }
    else if (stat == 1 && ring == 4) { // ME1a
        std::set<int> layersInME1b = layersInChamberHit[CSCDetId(endc,stat,1,cham,0)];
        std::set<int> layersInME11 = layersInChamberHit[chamberId];
        layersInME11.insert(layersInME1b.begin(),layersInME1b.end());
        nLayersInChamberHitForWireDigis = layersInME11.size();
    }
    else nLayersInChamberHitForWireDigis = layersInChamberHit[chamberId].size();

    unsigned int nLayersInChamberHitForStripDigis = layersInChamberHit[chamberId].size();
    
    if (nLayersInChamberHitForWireDigis < theLayersNeeded && nLayersInChamberHitForStripDigis < theLayersNeeded) continue;
    // skip bad chambers
    if ( !digitizeBadChambers_ && theConditions->isInBadChamber( CSCDetId(hitMapItr->first) ) ) continue;

    const CSCLayer * layer = findLayer(hitMapItr->first);
    const edm::PSimHitContainer & layerSimHits = hitMapItr->second;

    std::vector<CSCDetectorHit> newWireHits, newStripHits;
  
    LogTrace("CSCDigitizer") << "CSCDigitizer: found " << layerSimHits.size() <<" hit(s) in layer"
       << " E" << layer->id().endcap() << " S" << layer->id().station() << " R" << layer->id().ring()
       << " C" << layer->id().chamber() << " L" << layer->id().layer();

    // turn the edm::PSimHits into WireHits, using the WireHitSim
    {
      newWireHits.swap(theWireHitSim->simulate(layer, layerSimHits));
    }
    if(!newWireHits.empty()) {
      newStripHits.swap(theStripHitSim->simulate(layer, newWireHits));
    }

    // turn the hits into wire digis, using the electronicsSim
    if (nLayersInChamberHitForWireDigis >= theLayersNeeded) {
      theWireElectronicsSim->simulate(layer, newWireHits);
      theWireElectronicsSim->fillDigis(wireDigis);
      wireDigiSimLinks.insert( theWireElectronicsSim->digiSimLinks() );
    }  
    if (nLayersInChamberHitForStripDigis >= theLayersNeeded) {
      theStripElectronicsSim->simulate(layer, newStripHits);
      theStripElectronicsSim->fillDigis(stripDigis, comparators);
      stripDigiSimLinks.insert( theStripElectronicsSim->digiSimLinks() );
    }
  }

  // fill in the layers were missing from this chamber
  std::list<int> missingLayers = layersMissing(stripDigis);
  for(std::list<int>::const_iterator missingLayerItr = missingLayers.begin();
      missingLayerItr != missingLayers.end(); ++missingLayerItr)
  {
    const CSCLayer * layer = findLayer(*missingLayerItr);
    theStripElectronicsSim->fillMissingLayer(layer, comparators, stripDigis);
  }
}


std::list<int> CSCDigitizer::layersMissing(const CSCStripDigiCollection & stripDigis) const
{
  std::list<int> result;

  std::map<int, std::list<int> > layersInChamberWithDigi;
  for (CSCStripDigiCollection::DigiRangeIterator j=stripDigis.begin(); 
       j!=stripDigis.end(); j++) 
  {
    CSCDetId layerId((*j).first);
    // make sure the vector of digis isn't empty
    if((*j).second.first != (*j).second.second)
    {
      int chamberId = layerId.chamberId();
      layersInChamberWithDigi[chamberId].push_back(layerId.layer());
    }
 } 

  std::list<int> oneThruSix;
  for(int i = 1; i <=6; ++i)
    oneThruSix.push_back(i);

  for(std::map<int, std::list<int> >::iterator layersInChamberWithDigiItr = layersInChamberWithDigi.begin();
      layersInChamberWithDigiItr != layersInChamberWithDigi.end(); ++ layersInChamberWithDigiItr)
  {
    std::list<int> & layersHit = layersInChamberWithDigiItr->second;
    if (layersHit.size() < 6 && layersHit.size() >= theLayersNeeded) 
    {
      layersHit.sort();
      std::list<int> missingLayers(6);
      std::list<int>::iterator lastLayerMissing =
        set_difference(oneThruSix.begin(), oneThruSix.end(),
                       layersHit.begin(), layersHit.end(), missingLayers.begin());
      int chamberId = layersInChamberWithDigiItr->first;
      for(std::list<int>::iterator layerMissingItr = missingLayers.begin();
          layerMissingItr != lastLayerMissing; ++layerMissingItr)
      {
        // got from layer 1-6 to layer ID
        result.push_back(chamberId + *layerMissingItr); 
      }
    }
  }
  return result;
}


void CSCDigitizer::setMagneticField(const MagneticField * field) {
  theDriftSim->setMagneticField(field);
}


void CSCDigitizer::setStripConditions(CSCStripConditions * cond)
{
  theConditions = cond; // cache here
  theStripElectronicsSim->setStripConditions(cond); // percolate downwards
}


void CSCDigitizer::setParticleDataTable(const ParticleDataTable * pdt)
{
  theWireHitSim->setParticleDataTable(pdt);
}


void CSCDigitizer::setRandomEngine(CLHEP::HepRandomEngine& engine)
{
  theWireHitSim->setRandomEngine(engine);
  theWireElectronicsSim->setRandomEngine(engine);
  theStripElectronicsSim->setRandomEngine(engine);
  if(theNeutronReader) theNeutronReader->setRandomEngine(engine);
}


const CSCLayer * CSCDigitizer::findLayer(int detId) const {
  assert(theCSCGeometry != 0);
  const GeomDetUnit* detUnit = theCSCGeometry->idToDetUnit(CSCDetId(detId));
  if(detUnit == 0)
  {
    throw cms::Exception("CSCDigiProducer") << "Invalid DetUnit: " << CSCDetId(detId)
      << "\nPerhaps your signal or pileup dataset are not compatible with the current release?";
  }  
  return dynamic_cast<const CSCLayer *>(detUnit);
}

