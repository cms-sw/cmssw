#include "Geometry/RPCSimAlgo/interface/RPCRoll.h"
#include "Geometry/RPCSimAlgo/interface/RPCRollSpecs.h"
#include "SimMuon/RPCDigitizer/src/RPCSimSimple.h"

//NCA
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"

void
RPCSimSimple::simulate(const RPCRoll* roll,
		       const edm::PSimHitContainer& rpcHits )
{
  const Topology& topology=roll->specs()->topology();
  for (edm::PSimHitContainer::const_iterator _hit = rpcHits.begin();
       _hit != rpcHits.end(); ++_hit){

 
    // Here I hould check if the RPC are up side down;
    const LocalPoint& entr=_hit->entryPoint();
    const LocalPoint& exit=_hit->exitPoint();

    strips.insert(topology.channel(entr));  
    

  }
}


void
RPCSimSimple::fillDigis(int rollDetId, RPCDigiCollection& digis)
{
  for (std::set<int>::iterator i=strips.begin();
       i!=strips.end(); i++){
    RPCDigi rpcDigi(*i,0);
    //NCA
    digis.insertDigi(RPCDetId(rollDetId),rpcDigi);
  }
}

