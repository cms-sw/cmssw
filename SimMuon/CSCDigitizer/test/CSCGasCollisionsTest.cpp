// Very basic test of CSCGasCollisions
// Updated Mar-2015 now CSCGasCollisions has a dump flag.
// This runs despite error messages from CSCGasCollisions
// due to lack of real ParticleDataTable. (It doesn't matter
// because CSCGasCollisions continues to run assuming muon mass
// if table is bad or particle type does not exist in it.)

#include "SimMuon/CSCDigitizer/src/CSCGasCollisions.h"
#include "CLHEP/Random/JamesRandom.h"
#include <algorithm>
#include <numeric>
#include <functional>
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

int main() 
{

  // Set up pset param explicitly required by CSCGasCollisions.
  // Note that it has no effect because this stand-alone program
  // does not activate MessageLogger which now controls output
  // in CSCGasCollisions.
  edm::ParameterSet pset;
  pset.addUntrackedParameter<bool>("dumpGasCollisions", false); 
  pset.registerIt();

  CSCGasCollisions collisions (  pset );

  ParticleDataTable dummyTable;
  // let the code assume a muon
  collisions.setParticleDataTable(&dummyTable);

  CLHEP::HepJamesRandom engine;

  PSimHit simHit(LocalPoint(0.,0.,-0.5), LocalPoint(0.,0.,0.5),
                 4., 0., 0.000005, 13,
                 CSCDetId(1,1,1,1,1), 0, 0., 0., 0);

  /*
  PSimHit( const Local3DPoint& entry, const Local3DPoint& exit,
           float pabs, float tof, float eloss, int particleType,
           unsigned int detId, unsigned int trackId,
           float theta, float phi, unsigned short processType=0) :
  */

  int n = 100;
  int sumElectrons = 0;
  int sumClusters = 0;
  for(int i = 0; i < n; ++i) {
    std::vector<LocalPoint> clusters;
    std::vector<int> electrons;
    collisions.simulate(simHit, clusters, electrons, &engine);

    sumElectrons += std::accumulate(electrons.begin(), electrons.end(), 0);
    sumClusters += clusters.size();
  }

  std::cout << "Clusters: " << sumClusters/n << "  electrons: " << sumElectrons/n << std::endl;
  return 0;
}
