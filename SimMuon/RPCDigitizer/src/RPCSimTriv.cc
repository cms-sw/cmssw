#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCRollSpecs.h"
#include "SimMuon/RPCDigitizer/src/RPCSimTriv.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"



RPCSimTriv::RPCSimTriv(const edm::ParameterSet& config) : RPCSim(config){
}


void
RPCSimTriv::simulate(const RPCRoll* roll,
		       const edm::PSimHitContainer& rpcHits )
{
  const Topology& topology=roll->specs()->topology();
  for (edm::PSimHitContainer::const_iterator _hit = rpcHits.begin();
       _hit != rpcHits.end(); ++_hit){


    int type = _hit->particleType();
    if (type == 13 || type == -13){
      // Here I hould check if the RPC are up side down;
      const LocalPoint& entr=_hit->entryPoint();
      //    const LocalPoint& exit=_hit->exitPoint();

      strips.insert(topology.channel(entr)+1);  
    }
    

  }
}


