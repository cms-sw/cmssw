#include "SimMuon/CSCDigitizer/src/CSCWireHitSim.h"
#include "SimMuon/CSCDigitizer/src/CSCDriftSim.h"
#include "SimMuon/CSCDigitizer/src/CSCCrossGap.h"
#include "SimMuon/CSCDigitizer/src/CSCDetectorHit.h"
#include "SimMuon/CSCDigitizer/src/CSCGasCollisions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CSCSimAlgo/interface/CSCLayer.h"
#include "Geometry/CSCSimAlgo/interface/CSCLayerGeometry.h"
#include "Geometry/CSCSimAlgo/interface/CSCChamberSpecs.h"
#include "CLHEP/Units/SystemOfUnits.h"


CSCWireHitSim::CSCWireHitSim(CSCDriftSim* driftSim) 
: pDriftSim(driftSim),
  theGasIonizer( new CSCGasCollisions() ) 
{
}


CSCWireHitSim::~CSCWireHitSim() {
    delete theGasIonizer;
}


std::vector<CSCDetectorHit> &
CSCWireHitSim::simulate(const CSCLayer * layer, 
                        const edm::PSimHitContainer & simHits) 
{
  const CSCLayerGeometry * geom = layer->geometry(); 

  newWireHits.clear();
  for (edm::PSimHitContainer::const_iterator hitItr = simHits.begin();
       hitItr != simHits.end();  ++hitItr)
  {

    std::vector<LocalPoint> ionClusters 
      = getIonizationClusters(*hitItr, layer);

    for(int icl = 0; icl < int( ionClusters.size() ); ++icl) {

      // Drift the electrons in the cluster to the nearest wire...
      int nearestWire=geom->nearestWire(ionClusters[icl]);

      // The wire hit contains wire# and position measured _along the wire_
      // from where it intersects local y axis.

      newWireHits.push_back( 
          pDriftSim->getWireHit(ionClusters[icl], layer, nearestWire,
          *hitItr) );

    }
  } 
  return newWireHits;
}

std::vector<LocalPoint> 
CSCWireHitSim::getIonizationClusters(const PSimHit & simHit, 
     const CSCLayer * layer) {

  const LocalPoint & entryPoint = simHit.entryPoint();
  const LocalPoint & exitPoint  = simHit.exitPoint();

  LogDebug("CSCWireHitSim") << "CSCWireHitSim:" 
      << " type=" << simHit.particleType() 
      << " mom=" << simHit.pabs()
      << "\n Local entry " << entryPoint << " exit " << exitPoint;

  std::vector<LocalPoint> positions;
  std::vector<int> electrons;
  theGasIonizer->simulate( simHit, positions, electrons );

  //  std::vector<LocalPoint> results( positions ); //copy
  std::vector<LocalPoint> results; // start empty

  int j = 0;
  for( std::vector<LocalPoint>::const_iterator i = positions.begin(); 
                                         i != positions.end(); ++i ) {
    ++j;
    LocalPoint newPoint( *i );
    // some verification
    if(layer->geometry()->inside(newPoint) ) {
      // push the point for each electron at this point
      
      for( int ie = 1;  ie <= electrons[j-1]; ++ie ) {
        results.push_back(newPoint);
      }
    }
  }
  LogDebug("CSCWireHitSim") << "MEWHS: there are " << results.size()
     << " clusters identified with each electron.";
  return results;
}


