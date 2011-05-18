#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/ClusteringModule.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <sstream>


using namespace l1slhc;
using namespace std;

ClusteringModule::ClusteringModule():
  L1CaloTowerModule()
{}

ClusteringModule::ClusteringModule(const L1CaloTriggerSetup& rcd):
  L1CaloTowerModule(rcd)
{
}

ClusteringModule::~ClusteringModule()
{
}


void
ClusteringModule::populateLattice(const edm::Handle<L1CaloTowerCollection>& towers)
{
  //Loop On the Hits and see which Hits are inside the region. Then Populate the region


  //Do it for ECAL
  if(towers->size()>0)
      for(L1CaloTowerCollection::const_iterator Tower = towers->begin();Tower!=towers->end();++Tower)
	if(Tower->E()+Tower->H()>0)
	  for(std::map<int,std::pair< int,int > >::iterator it = s.geoMap_.begin();it!=s.geoMap_.end();++it)
	    if(it->second.first==Tower->iEta() && it->second.second==Tower->iPhi())
	      {
		L1CaloTowerRef ref(towers,Tower-towers->begin());
		lattice_[it->first] = ref;
	      }
  
}

void
ClusteringModule::clusterize(L1CaloClusterCollection& clusters,const edm::Handle<L1CaloTowerCollection>& towers)
{
  //I am doing it the slow way because I want to simulate the Circuit!!
  reset();

  //Populate the lattice with ECAL
  populateLattice(towers);


  //Perform the Sliding Window Algorithm
  //The last row and  column are repeated  for the overlap
  for(int phi =s.phiMin() ;phi<=s.phiMax();++phi)
    for(int eta =s.etaMin() ;eta<=s.etaMax();++eta)
    {
      
      //Get lattice Point
      int bin = s.getBin(eta,phi);
      //Create a cluster
      std::pair<int,int> p  =s.geoMap_[bin];
      L1CaloCluster cl(p.first,p.second);
      bool fineGrain=false;
      int ECALE=0;
      int E=0;

      int LEADTOWER=0;

      for( int bin_eta = eta; bin_eta<=eta+1;++bin_eta)
	for( int bin_phi = phi; bin_phi<=phi+1;++bin_phi) {
	  int bin_in = s.getBin(bin_eta,bin_phi);
	  if(isValid(bin_in))
	    {
	      //Skip over fine grain bit calculation if desired
	      if (s.fineGrainPass()==1) {
		fineGrain=false;
	      } else {
		fineGrain = fineGrain||objectAt(bin_in)->EcalFG();
	      }E+=objectAt(bin_in)->E()+objectAt(bin_in)->H();
	      ECALE+=objectAt(bin_in)->E();
	      cl.addConstituent(objectAt(bin_in));
	      if(objectAt(bin_in)->E()+objectAt(bin_in)->H()>=LEADTOWER)
		LEADTOWER=objectAt(bin_in)->E()+objectAt(bin_in)->H();
	    }
	}     
      //Calculate Electron Cut
      int electronValue =(int)(100.*((double)ECALE)/((double)E));
      cl.setLeadTower( LEADTOWER >= s.seedTowerThr());
      //Save it in the Cluster
      cl.setEGammaValue(electronValue);
      //Electron Bit Decision
      bool lowPtElectron  = (cl.E()<=s.electronThr()[1] && electronValue>s.electronThr()[0]);
      bool highPtElectron = ( cl.E()>s.electronThr()[1] && electronValue>(s.electronThr()[0]-(int)(((double)s.electronThr()[2])/10.)*(cl.E()-s.electronThr()[1])));
      
      if(lowPtElectron ||highPtElectron)
	cl.setEGamma(true);
      else
	cl.setEGamma(false);
      //FineGrain bit
      cl.setFg(fineGrain);
      
      if(cl.E()>0)
	clusters.push_back(cl);

    }
}


