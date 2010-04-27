#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/IsolationModule.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <sstream>
using namespace l1slhc;
using namespace std;


IsolationModule::IsolationModule():
  L1CaloClusterModule()
{

}

IsolationModule::IsolationModule(const L1CaloTriggerSetup& rcd):
  L1CaloClusterModule(rcd)
{

}


IsolationModule::~IsolationModule()
{

}


void
IsolationModule::populateLattice(const edm::Handle<l1slhc::L1CaloClusterCollection>& clusters)
{
  if(clusters->size()>0)
  for(l1slhc::L1CaloClusterCollection::const_iterator cl = clusters->begin();cl!=clusters->end();++cl)
        for(std::map<int,std::pair< int,int > >::iterator it = s.geoMap_.begin();it!=s.geoMap_.end();++it)
	  if(it->second.first==cl->iEta() && it->second.second==cl->iPhi())
          {
	    L1CaloClusterRef ref(clusters,cl-clusters->begin());
	    lattice_[it->first] = ref;
          }
}


void
IsolationModule::isoDeposits(const edm::Handle<l1slhc::L1CaloClusterCollection>& clusters,l1slhc::L1CaloClusterCollection& isoClusters)
{
  //I am doing it the slow way because I want to simulate the Circuit!!
  
  //Reset and Populate the lattice
  reset();
  populateLattice(clusters);

  //Loop on the Lattice
  for(int bin_phi=s.phiMin();bin_phi<=s.phiMax();++bin_phi)
    for(int bin_eta=0;bin_eta<=s.etaMax()-1;++bin_eta)
      {
	//Look if there is a cluster here and if the cluster is central (not pruned)
	
	int binOrigin = s.getBin(bin_eta,bin_phi);
	if(isValid(binOrigin))
	  if(objectAt(binOrigin)->isCentral())
	    {
	      int nClustersEG=0;
	      int nClustersTau=0;
	      L1CaloCluster origin = *objectAt(binOrigin);
	      //There is a cluster here:Calculate isoDeposits
	      for(int phi=bin_phi-s.nIsoTowers();phi<=bin_phi+s.nIsoTowers()+1;++phi)
		for(int eta=bin_eta-s.nIsoTowers();eta<=bin_eta+s.nIsoTowers()+1;++eta)
		  if(!(eta==bin_eta && phi==bin_phi))
		    {
		      //Take this cluster
		      int bin = s.getBin(eta,phi);
		      //If neighbor exists
		      if(isValid(bin))
			{
			  if(objectAt(bin)->E() >= s.isoThr()[0])
			    {
			      nClustersEG++;
			    }
			  if(objectAt(bin)->E() >= s.isoThr()[1])
			    {
			      nClustersTau++;
			    }
			}
		    }
	      

	      origin.setIsoClusters(nClustersEG,nClustersTau);
	      
	      
	      //Calculate Bits Tau isolation / electron Isolation
	      if(isoLookupTable(nClustersEG,s.isolationE()[0],s.isolationE()[1],origin.E()))
		{
		  origin.setIsoEG(true);
		}
	      
	      //Add the LUT inputs 
	      
	      if(isoLookupTable(nClustersTau,s.isolationT()[0],s.isolationT()[1],origin.E()))
		{
		  origin.setIsoTau(true);
		}
	      isoClusters.push_back(origin);
	    }
      }
}



bool 
IsolationModule::isoLookupTable(int clusters, int coeffA,int coeffB,int E) //takes as input the # Clusters the isolation coefficients
{
  bool decision=false;
  if(E>=0&&E<16)
    {
      //      compressedEnergy=0;
      if(clusters<=(int)(coeffA+(double)(coeffB*8)/1000.))
	decision=true;
    }
  else if(E>=16&&E<32)
    {
      //      compressedEnergy=1;
      if(clusters<=(int)(coeffA+(double)(coeffB*24)/1000.))
	decision=true;
    }
  else if(E>=32&&E<48)
    {
      //      compressedEnergy=2;
      if(clusters<=(int)(coeffA+(double)(coeffB*40)/1000.))
	decision=true;
    }

  else if(E>=48&&E<64)
    {
      //      compressedEnergy=3;
      if(clusters<=(int)(coeffA+(double)(coeffB*56)/1000.))
	decision=true;
    }

  else if(E>=64&&E<80)
    {
      //      compressedEnergy=4;
      if(clusters<=(int)(coeffA+(double)(coeffB*72)/1000.))
	decision=true;
    }

  else if(E>=80&&E<96)
    {
      //      compressedEnergy=5;
      if(clusters<=(int)(coeffA+(double)(coeffB*88)/1000.))
	decision=true;
    }

  else if(E>=96&&E<112)
    {
      //      compressedEnergy=6;
      if(clusters<=(int)(coeffA+(double)(coeffB*104)/1000.))


	decision=true;
    }
  else if(E>=112&&E<128)
    {
      //      compressedEnergy=7;
      if(clusters<=(int)(coeffA+(double)(coeffB*120)/1000.))
	decision=true;
    }
  else if(E>=128&&E<144)
    {
      //      compressedEnergy=8;
      if(clusters<=(int)(coeffA+(double)(coeffB*136)/1000.))
	decision=true;
    }
  else if(E>=144&&E<160)
    {
      //      compressedEnergy=9;
      if(clusters<=(int)(coeffA+(double)(coeffB*152)/1000.))
	decision=true;
    }
  else if(E>=160)
    {
      //      compressedEnergy=10;
	decision=true;
    } 
  /*.........................*/
    
  return decision;

}

