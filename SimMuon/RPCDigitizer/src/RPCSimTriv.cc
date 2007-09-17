#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCRollSpecs.h"
#include "SimMuon/RPCDigitizer/src/RPCSimTriv.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CLHEP/Random/RandomEngine.h"
#include "CLHEP/Random/RandFlat.h"
#include <cmath>
#include <utility>
#include <map>

#include "CLHEP/config/CLHEP.h"
#include "CLHEP/Random/Random.h"
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandPoissonQ.h"


RPCSimTriv::RPCSimTriv(const edm::ParameterSet& config) : RPCSim(config){

  rate=config.getParameter<double>("Rate");
  nbxing=config.getParameter<int>("Nbxing");
  gate=config.getParameter<double>("Gate");

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
RPCSimTriv::simulate(const RPCRoll* roll,
		       const edm::PSimHitContainer& rpcHits,const RPCGeometry* geo )
{

  _rpcSync->setGeometry(geo);
  _rpcSync->setReadOutTime(geo);

  const Topology& topology=roll->specs()->topology();
  for (edm::PSimHitContainer::const_iterator _hit = rpcHits.begin();
       _hit != rpcHits.end(); ++_hit){

    int type = _hit->particleType();
    if (type == 13 || type == -13){
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
}

void RPCSimTriv::simulateNoise(const RPCRoll* roll)
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
  
  double ave = rate*nbxing*gate*area*1.0e-9;
  
  N_hits = RandPoissonQ::shoot(ave);
  for (int i = 0; i < N_hits; i++ ){
      int strip = RandFlat::shootInt(nstrips);
      int time_hit = static_cast<int>(RandFlat::shoot((nbxing*gate))/gate);
      std::pair<int, int> digi(strip,time_hit);
      strips.insert(digi);
  }
}
