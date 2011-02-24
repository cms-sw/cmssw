#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCRollSpecs.h"
#include "SimMuon/RPCDigitizer/src/RPCSimTriv.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "SimMuon/RPCDigitizer/src/RPCSimSetUp.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CLHEP/Random/RandomEngine.h"
#include "CLHEP/Random/RandFlat.h"
#include <cmath>
#include <utility>
#include <map>

//#include "CLHEP/config/CLHEP.h"
#include "CLHEP/Random/Random.h"
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandPoissonQ.h"


RPCSimTriv::RPCSimTriv(const edm::ParameterSet& config) : RPCSim(config){

  rate=config.getParameter<double>("Rate");
  nbxing=config.getParameter<int>("Nbxing");
  gate=config.getParameter<double>("Gate");

  _rpcSync = new RPCSynchronizer(config);
}

void RPCSimTriv::setRandomEngine(CLHEP::HepRandomEngine& eng){
  flatDistribution1 = new CLHEP::RandFlat(eng);
  flatDistribution2 = new CLHEP::RandFlat(eng);
  poissonDistribution = new CLHEP::RandPoissonQ(eng);
  _rpcSync->setRandomEngine(eng);
}

RPCSimTriv::~RPCSimTriv(){
  delete flatDistribution1;
  delete flatDistribution2;
  delete poissonDistribution;
  delete _rpcSync;
}

void
RPCSimTriv::simulate(const RPCRoll* roll,
		       const edm::PSimHitContainer& rpcHits)
{

  //_rpcSync->setRPCSimSetUp(getRPCSimSetUp());
  theRpcDigiSimLinks.clear();
  theDetectorHitMap.clear();
  theRpcDigiSimLinks = RPCDigiSimLinks(roll->id().rawId());

  const Topology& topology=roll->specs()->topology();
  for (edm::PSimHitContainer::const_iterator _hit = rpcHits.begin();
       _hit != rpcHits.end(); ++_hit){

    int type = _hit->particleType();
    if (type == 13 || type == -13){
      // Here I hould check if the RPC are up side down;
      const LocalPoint& entr=_hit->entryPoint();
      //int time_hit = _rpcSync->getSimHitBx(&(*_hit));
      // please keep hit time always 0 for this model 
      int time_hit = 0;
      std::pair<int, int> digi(topology.channel(entr)+1,
			       time_hit);

      theDetectorHitMap.insert(DetectorHitMap::value_type(digi,&(*_hit)));
      strips.insert(digi);
    }
  }
}


void RPCSimTriv::simulateNoise(const RPCRoll* roll)
{
  // plase keep it empty for this model
  return;
}

