#include "SimMuon/CSCDigitizer/src/CSCWireHitSim.h"
#include "SimMuon/CSCDigitizer/src/CSCDriftSim.h"
#include "SimMuon/CSCDigitizer/src/CSCGasCollisions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CSCGeometry/interface/CSCLayer.h"
#include "Geometry/CSCGeometry/interface/CSCLayerGeometry.h"

#include "CLHEP/Random/RandFlat.h"

CSCWireHitSim::CSCWireHitSim(CSCDriftSim* driftSim, const edm::ParameterSet & p) 
: theDriftSim(driftSim),
  theGasIonizer( new CSCGasCollisions( p ) ) ,
  theNewWireHits()
{
}


CSCWireHitSim::~CSCWireHitSim() {
  delete theGasIonizer;
}


std::vector<CSCDetectorHit> &
CSCWireHitSim::simulate(const CSCLayer * layer, 
                        const edm::PSimHitContainer & simHits,
                        CLHEP::HepRandomEngine* engine)
{
  const CSCLayerGeometry * geom = layer->geometry(); 

  theNewWireHits.clear();
  for (edm::PSimHitContainer::const_iterator hitItr = simHits.begin();
       hitItr != simHits.end();  ++hitItr)
  {

    std::vector<LocalPoint> ionClusters 
      = getIonizationClusters(*hitItr, layer, engine);

    unsigned nClusters = ionClusters.size();
    theNewWireHits.reserve(theNewWireHits.size()+nClusters);

    for(unsigned icl = 0; icl < nClusters; ++icl) {

      // Drift the electrons in the cluster to the nearest wire...
      int nearestWire=geom->nearestWire(ionClusters[icl]);

      // The wire hit contains wire# and position measured _along the wire_
      // from where it intersects local y axis.

      theNewWireHits.push_back( 
          theDriftSim->getWireHit(ionClusters[icl], layer, nearestWire,
                                  *hitItr, engine) );

    }
  } 
  return theNewWireHits;
}

std::vector<LocalPoint> 
CSCWireHitSim::getIonizationClusters(const PSimHit & simHit, 
                                     const CSCLayer * layer,
                                     CLHEP::HepRandomEngine* engine)
{
  const LocalPoint & entryPoint = simHit.entryPoint();
  const LocalPoint & exitPoint  = simHit.exitPoint();

  LogTrace("CSCWireHitSim") << "CSCWireHitSim:" 
      << " type=" << simHit.particleType() 
      << " mom=" << simHit.pabs()
      << "\n Local entry " << entryPoint << " exit " << exitPoint;

  std::vector<LocalPoint> positions;
  std::vector<int> electrons;
  theGasIonizer->simulate( simHit, positions, electrons, engine );

  std::vector<LocalPoint> results; // start empty

  int j = 0;
  for( std::vector<LocalPoint>::const_iterator pointItr = positions.begin(); 
                                         pointItr != positions.end(); ++pointItr ) 
  {
    ++j;
    // some verification
    if(layer->geometry()->inside(*pointItr) ) {
      // push the point for each electron at this point
      
      for( int ie = 1;  ie <= electrons[j-1]; ++ie ) {
        // probability of getting attached
        float f_att = 0.5;
        if(CLHEP::RandFlat::shoot(engine) > f_att) {
          results.push_back(*pointItr);
        }
      }
    }
  }
  LogTrace("CSCWireHitSim") << "CSCWireHitSim: there are " << results.size()
     << " clusters identified with each electron.";
  return results;
}


void CSCWireHitSim::setParticleDataTable(const ParticleDataTable * pdt)
{
  theGasIonizer->setParticleDataTable(pdt);
}
