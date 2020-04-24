#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCRollSpecs.h"
#include "SimMuon/RPCDigitizer/src/RPCSimSimple.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"

#include <cmath>

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandPoissonQ.h"

#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <utility>
#include <map>

RPCSimSimple::RPCSimSimple(const edm::ParameterSet& config) : RPCSim(config){

  rate=config.getParameter<double>("Rate");
  nbxing=config.getParameter<int>("Nbxing");
  gate=config.getParameter<double>("Gate");

  _rpcSync = new RPCSynchronizer(config);
}

RPCSimSimple::~RPCSimSimple(){
  delete _rpcSync;
}


void
RPCSimSimple::simulate(const RPCRoll* roll,
		       const edm::PSimHitContainer& rpcHits,
                       CLHEP::HepRandomEngine* engine)
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
    int time_hit = _rpcSync->getSimHitBx(&(*_hit), engine);
    //    const LocalPoint& exit=_hit->exitPoint();
	
    std::pair<int, int> digi(topology.channel(entr)+1,
			     time_hit);

    theDetectorHitMap.insert(DetectorHitMap::value_type(digi,&(*_hit)));
    strips.insert(digi);
  }
}


void RPCSimSimple::simulateNoise(const RPCRoll* roll,
                                 CLHEP::HepRandomEngine* engine)
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

  CLHEP::RandPoissonQ randPoissonQ(*engine, ave);
  N_hits = randPoissonQ.fire();

  for (int i = 0; i < N_hits; i++ ){
    int strip = static_cast<int>(CLHEP::RandFlat::shoot(engine, 1, nstrips));
    int time_hit;
    time_hit = (static_cast<int>(CLHEP::RandFlat::shoot(engine, (nbxing*gate)/gate))) - nbxing/2;
    std::pair<int, int> digi(strip,time_hit);
    strips.insert(digi);
  }
}
