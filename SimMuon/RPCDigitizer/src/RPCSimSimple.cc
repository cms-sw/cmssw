#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCRollSpecs.h"
#include "SimMuon/RPCDigitizer/src/RPCSimSimple.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CLHEP/Random/RandomEngine.h"
#include "CLHEP/Random/RandFlat.h"
#include <cmath>

#include<cstring>
#include<iostream>
#include<fstream>
#include<string>
#include<vector>
#include<stdlib.h>
#include <utility>
#include <map>

RPCSimSimple::RPCSimSimple(const edm::ParameterSet& config) : RPCSim(config){

  edm::Service<edm::RandomNumberGenerator> rng;
  if ( ! rng.isAvailable()) {
    throw cms::Exception("Configuration")
      << "RPCDigitizer requires the RandomNumberGeneratorService\n"
      "which is not present in the configuration file.  You must add the service\n"
      "in the configuration file or remove the modules that require it.";
  }
  
  _rpcSync = new RPCSynchronizer(config);

  rndEngine = &(rng->getEngine());
  flatDistribution = new CLHEP::RandFlat(rndEngine);

}

void
RPCSimSimple::simulate(const RPCRoll* roll,
		       const edm::PSimHitContainer& rpcHits,
		       const RPCGeometry* geo )
{

  _rpcSync->setGeometry(geo);
  _rpcSync->setReadOutTime(geo);

  const Topology& topology=roll->specs()->topology();
  for (edm::PSimHitContainer::const_iterator _hit = rpcHits.begin();
       _hit != rpcHits.end(); ++_hit){

 
    // Here I hould check if the RPC are up side down;
    const LocalPoint& entr=_hit->entryPoint();
    //    const LocalPoint& exit=_hit->exitPoint();
	
    std::pair<int, int> digi(topology.channel(entr)+1,
			     _rpcSync->getDigiBx(&(*_hit), 
						 topology.channel(entr)+1, 
						 topology.channel(entr)+1));
    //	std::cout<<"STRIP: "<<*i<<"  "<<"BX: "<<bx<<std::endl;
    strips.insert(digi);

  }
}


