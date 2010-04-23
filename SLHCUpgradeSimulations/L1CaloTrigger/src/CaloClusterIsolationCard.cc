#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/CaloClusterIsolationCard.h"
#include <sstream>
using namespace l1slhc;
using namespace std;


CaloClusterIsolationCard::CaloClusterIsolationCard()
{

}

CaloClusterIsolationCard::CaloClusterIsolationCard(const L1CaloTriggerSetup& rcd)
{
  s = rcd;
}


CaloClusterIsolationCard::~CaloClusterIsolationCard()
{

}


void
CaloClusterIsolationCard::populateLattice(const l1slhc::L1CaloClusterCollection& clusters)
{
  if(clusters.size()>0)
  for(l1slhc::L1CaloClusterCollection::const_iterator cl = clusters.begin();cl!=clusters.end();++cl)
        for(std::map<int,std::pair< int,int > >::iterator it = s.geoMap_.begin();it!=s.geoMap_.end();++it)
            if(it->second.first==cl->iEta() && it->second.second==cl->iPhi())
          {
	    lattice_[it->first] = *cl;
          }
}


void
CaloClusterIsolationCard::isoDeposits(const l1slhc::L1CaloClusterCollection& clusters,l1slhc::L1CaloClusterCollection& isoClusters)
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
	    std::map<int,L1CaloCluster>::iterator iter;
	    iter = lattice_.find(s.getBin(bin_eta,bin_phi));
	    if(iter!=lattice_.end())
	      if(iter->second.isCentral())
		{
		  int nClustersEG=0;
		  int nClustersTau=0;
		  L1CaloCluster origin = iter->second;
		  
		  //There is a cluster here:Calculate isoDeposits
		  for(int phi=bin_phi-s.nIsoTowers();phi<=bin_phi+s.nIsoTowers();++phi)
		    for(int eta=bin_eta-s.nIsoTowers();eta<=bin_eta+s.nIsoTowers();++eta)
		      if(!(eta==bin_eta && phi==bin_phi))
			{
			  //Take this cluster
			  int bin = s.getBin(eta,phi);
			  std::map<int,L1CaloCluster>::iterator iter2= lattice_.find(bin);
			  //If neighbor exists
			  if(iter2!=lattice_.end())
			    {
			      if(iter2->second.E() >= s.isoThr()[0])
				{
				  nClustersEG++;
				}
			      if(iter2->second.E() >= s.isoThr()[1])
				{
				  nClustersTau++;
				}
			    }
			}


		  origin.setIsoClusters(nClustersEG,nClustersTau);
		  
		  //Calculate Bits Tau isolation / electron Isolation
		  if(nClustersEG <=(int)( s.isolationE()[0]+(double)(s.isolationE()[1]*origin.E())/1000.))
		    {
		      origin.setIsoEG(true);
		    }
		  //For the tau check if it isolated //

		  if(nClustersTau <=(int)( s.isolationT()[0]+(double)(s.isolationT()[1]*origin.E())/1000.))
		    {
		      origin.setIsoTau(true);
		    }
		  isoClusters.push_back(origin);
		}
	  }
}







