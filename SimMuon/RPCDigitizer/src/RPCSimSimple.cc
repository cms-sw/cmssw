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

//#include "CLHEP/config/CLHEP.h"
#include "CLHEP/Random/Random.h"
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandPoissonQ.h"


#include<cstring>
#include<iostream>
#include<string>
#include<vector>
#include<stdlib.h>
#include <utility>
#include <map>

RPCSimSimple::RPCSimSimple(const edm::ParameterSet& config) : RPCSim(config){

  rate=config.getParameter<double>("Rate");
  nbxing=config.getParameter<int>("Nbxing");
  gate=config.getParameter<double>("Gate");

//   edm::Service<edm::RandomNumberGenerator> rng;
//   if ( ! rng.isAvailable()) {
//     throw cms::Exception("Configuration")
//       << "RPCDigitizer requires the RandomNumberGeneratorService\n"
//       "which is not present in the configuration file.  You must add the service\n"
//       "in the configuration file or remove the modules that require it.";
//   }
  
  _rpcSync = new RPCSynchronizer(config);

//   CLHEP::HepRandomEngine& rndEngine = rng->getEngine();
//   flatDistribution_ = new CLHEP::RandFlat(rndEngine);
}


RPCSimSimple::~RPCSimSimple(){
  delete _rpcSync;
}


void
RPCSimSimple::simulate(const RPCRoll* roll,
		       const edm::PSimHitContainer& rpcHits)
{
  _rpcSync->setRPCSimSetUp(getRPCSimSetUp());
  theRpcDigiSimLinks.clear();
  theDetectorHitMap.clear();
  theRpcDigiSimLinks = RPCDigiSimLinks(roll->id().rawId());

  const Topology& topology=roll->specs()->topology();
  for (edm::PSimHitContainer::const_iterator _hit = rpcHits.begin();
       _hit != rpcHits.end(); ++_hit){

 
    // Here I hould check if the RPC are up side down;
    const LocalPoint& entr=_hit->entryPoint();
    int time_hit = _rpcSync->getSimHitBx(&(*_hit));
    //    const LocalPoint& exit=_hit->exitPoint();
	
    std::pair<int, int> digi(topology.channel(entr)+1,
			     time_hit);

    theDetectorHitMap.insert(DetectorHitMap::value_type(digi,&(*_hit)));
    strips.insert(digi);
  }
}


void RPCSimSimple::simulateNoise(const RPCRoll* roll)
{

  RPCDetId rpcId = roll->id();
  int nstrips = roll->nstrips();
  double area = 0.0;
  
  if ( rpcId.region() == 0 )
    {
      const RectangularStripTopology* top_ = dynamic_cast<const
	RectangularStripTopology*>(&(roll->topology()));
      float xmin = (top_->localPosition(0.)).x();
      float xmax = (top_->localPosition((float)roll->nstrips())).x();
      float striplength = (top_->stripLength());
      area = striplength*(xmax-xmin);
    }
  else
    {
      const TrapezoidalStripTopology* top_=dynamic_cast<const TrapezoidalStripTopology*>(&(roll->topology()));
      float xmin = (top_->localPosition(0.)).x();
      float xmax = (top_->localPosition((float)roll->nstrips())).x();
      float striplength = (top_->stripLength());
      area = striplength*(xmax-xmin);
    }
  
  //Defining a new engine local to this method for the two distributions defined below
  edm::Service<edm::RandomNumberGenerator> rnd;
  if ( ! rnd.isAvailable()) {
    throw cms::Exception("Configuration")
      << "RPCDigitizer requires the RandomNumberGeneratorService\n"
      "which is not present in the configuration file.  You must add the service\n"
      "in the configuration file or remove the modules that require it.";
  }
  
  CLHEP::HepRandomEngine& engine = rnd->getEngine();
  //Taking the flatDistribution out of the for loop since it does not depend on
  //loop variables and deleting it outside or the larger for loop
  //Renaming it since it has same name as the one defined in the constructor and
  //used in getClSize and simulate methods.
  double ave = rate*nbxing*gate*area*1.0e-9;
  CLHEP::RandPoissonQ poissonDistribution(engine, ave);
  N_hits = poissonDistribution.fire();
  CLHEP::RandFlat flatDistribution1(engine, 1, nstrips);
  CLHEP::RandFlat  flatDistribution2(engine, (nbxing*gate)/gate);

  for (int i = 0; i < N_hits; i++ ){
    int strip = static_cast<int>(flatDistribution1.fire());
    int time_hit;
    time_hit = (static_cast<int>(flatDistribution2.fire())) - nbxing/2;
    std::pair<int, int> digi(strip,time_hit);
    strips.insert(digi);
  }
}
