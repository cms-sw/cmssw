#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/RegionModule.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <sstream>


using namespace l1slhc;
using namespace std;

RegionModule::RegionModule():
  L1CaloTowerModule()
{
}

RegionModule::RegionModule(const L1CaloTriggerSetup& rcd):
  L1CaloTowerModule(rcd)
{

}
RegionModule::~RegionModule()
{
}


void
RegionModule::populateLattice(const edm::Handle<L1CaloTowerCollection>& towers)
{
  //Loop On the Hits and see which Hits are inside the region. Then Populate the region

  //Do it for ECAL
  if(towers->size()>0)
      for(L1CaloTowerCollection::const_iterator Tower = towers->begin();Tower!=towers->end();++Tower)
	if(Tower->E()>0)
        for(std::map<int,std::pair< int,int > >::iterator it = s.geoMap_.begin();it!=s.geoMap_.end();++it)
	  if(it->second.first==Tower->iEta() && it->second.second==Tower->iPhi())
	    {
	      L1CaloTowerRef ref(towers,Tower-towers->begin());
	      lattice_[it->first] = ref;
	    }
}

void
RegionModule::clusterize(L1CaloRegionCollection& regions,const edm::Handle<L1CaloTowerCollection>& towers)
{
  //I am doing it the slow way because I want to simulate the Circuit!!
  reset();
  


  //Populate the lattice with ECAL/HCAL
  populateLattice(towers);


  //Perform the Sliding Window Algorithm
  //The last row and  column are repeated  for the overlap
  for(int phi =s.phiMin() ;phi<=s.phiMax()-3;phi+=4)
    for(int eta =s.etaMin() ;eta<=s.etaMax()-3;eta+=4) {
      int E=0;
      for(int p=phi;p<phi+4;++p)
	for(int e=eta;e<eta+4;++e)
	  {
	    int bin1 = s.getBin(e,p);
	    if(isValid(bin1))
	      {
		E+=objectAt(bin1)->E()+objectAt(bin1)->H();
	      }
	  }


      //Get lattice Point
      int bin = s.getBin(eta,phi);
      //Create a region
      std::pair<int,int> p  =s.geoMap_[bin];
      L1CaloRegion region(p.first,p.second,E);
      if(E>0)
	regions.push_back(region);


    }

}
	
