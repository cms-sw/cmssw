#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/CaloClusterFilteringCard.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <sstream>


using namespace l1slhc;
using namespace std;


CaloClusterFilteringCard::CaloClusterFilteringCard()
{

}

CaloClusterFilteringCard::CaloClusterFilteringCard(const L1CaloTriggerSetup& rcd)
{
  s = rcd;
}


CaloClusterFilteringCard::~CaloClusterFilteringCard()
{

}

void
CaloClusterFilteringCard::populateLattice(const l1slhc::L1CaloClusterCollection& clusters)
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
CaloClusterFilteringCard::cleanClusters(const l1slhc::L1CaloClusterCollection& clusters,l1slhc::L1CaloClusterCollection& cleanClusters)
{
  //I am doing it the slow way because I want to simulate the Circuit!!

  //Reset and Populate the lattice
  reset();
  populateLattice(clusters);
  int totalET=0;

  //Loop on the Lattice
      for(int bin_phi=s.phiMin();bin_phi<=s.phiMax();++bin_phi)
	for(int bin_eta=0;bin_eta<=s.etaMax()-1;++bin_eta)
          {
	    //Look if there is a cluster here
	    std::map<int,L1CaloCluster>::iterator iter;
	    iter = lattice_.find(s.getBin(bin_eta,bin_phi));
	    if(iter!=lattice_.end())
	      {
		std::ostringstream pattern;
		pattern << "OF:New Cluster Filter\n";

		//There is cluster- loop on the neighbors
		std::map<int,L1CaloCluster>::iterator iter2;
		L1CaloCluster origin = iter->second;
		int bin=-1;

		pattern << "OF:Initial Cluster: "<< origin.towerE(0) <<
		  " "<< origin.towerE(1) << " " << origin.towerE(2) <<" "<<
		  origin.towerE(3) <<"\n";

		//Calculate the initial energy since THIS you will compare!
		int originE = origin.E();
		std::pair<int,int> posIn =  calculateClusterPosition(origin);

		//Set central bit
		bool central=true;


		//right
		bin = s.getBin(bin_eta+1,bin_phi);
		iter2 = lattice_.find(bin);
		//If neighbor exists
		if(iter2!=lattice_.end())
		  {
		    L1CaloCluster neighbor = iter2->second;
		    
		    pattern << "OF:E Cluster: "<< neighbor.towerE(0) <<
	       " "<< neighbor.towerE(1) << " " << neighbor.towerE(2) <<" "<<
		      neighbor.towerE(3) <<"\n";

		    //Compare the energies and prune if the neighbor has higher Et
		    if(originE <= neighbor.E())
		      {
			origin.setTower(1,0);
			origin.setTower(3,0);
			central=false;
		      }
		  }
		else
		  {
		    pattern << "OF:E Cluster: 0 0 0 0\n";
		  }



		//right-down
		bin = s.getBin(bin_eta+1,bin_phi+1);
		iter2 = lattice_.find(bin);
		//If neighbor exists
		if(iter2!=lattice_.end())
		  {
		    L1CaloCluster neighbor = iter2->second;
		    pattern << "OF:SE Cluster: "<< neighbor.towerE(0) <<
		      " "<< neighbor.towerE(1) << " " << neighbor.towerE(2) <<" "<<
		      neighbor.towerE(3) <<"\n";
		    //Compare the energies and prune if the neighbor has higher Et
		    if(originE <= neighbor.E())
		      {
			origin.setTower(3,0);
			central=false;
			
		      }
		  }
		else
		  {
		    pattern << "OF:SE Cluster: 0 0 0 0\n";
		  }
		
		//down
		bin = s.getBin(bin_eta,bin_phi+1);
		iter2 = lattice_.find(bin);
		//If neighbor exists
		if(iter2!=lattice_.end())
		  {
		    L1CaloCluster neighbor = iter2->second;
		    pattern << "OF:S Cluster: "<< neighbor.towerE(0) <<
		      " "<< neighbor.towerE(1) << " " << neighbor.towerE(2) <<" "<<
		      neighbor.towerE(3) <<"\n";
		    
		    //Compare the energies and prune if the neighbor has higher Et
		    if(originE <= neighbor.E())
		      {
			origin.setTower(2,0);
			origin.setTower(3,0);
			central=false;
			
		      }
		  }
		else
		  {
		    pattern << "OF:S Cluster: 0 0 0 0\n";
		  }
		
		//down-left
		bin = s.getBin(bin_eta-1,bin_phi+1);
		iter2 = lattice_.find(bin);
		//If neighbor exists
		if(iter2!=lattice_.end())
		  {
		    L1CaloCluster neighbor = iter2->second;
		    
		    pattern << "OF:SW Cluster: "<< neighbor.towerE(0) <<
		      " "<< neighbor.towerE(1) << " " << neighbor.towerE(2) <<" "<<
		      neighbor.towerE(3) <<"\n";
		    
		    //Compare the energies and prune if the neighbor has higher Et
		    if(originE <= neighbor.E())
		      {
			origin.setTower(2,0);
			central=false;
			
		      }
		  }
		else
		  {
		    pattern << "OF:SW Cluster: 0 0 0 0\n";
		  }

		//left
		bin = s.getBin(bin_eta-1,bin_phi);
		iter2 = lattice_.find(bin);
		//If neighbor exists
		if(iter2!=lattice_.end())
		  {
		    L1CaloCluster neighbor = iter2->second;
		    
		    pattern << "OF:W Cluster: "<< neighbor.towerE(0) <<
		      " "<< neighbor.towerE(1) << " " << neighbor.towerE(2) <<" "<<
		      neighbor.towerE(3) <<"\n";
		    
		    //Compare the energies and prune if the neighbor has higher Et
		    if(originE < neighbor.E())
		      {
			origin.setTower(0,0);
			origin.setTower(2,0);
			central=false;
			
		      }
		  }
		else
		  {
		    pattern << "OF:W Cluster: 0 0 0 0\n";
		  }
		
		
		//left-up
		bin = s.getBin(bin_eta-1,bin_phi-1);
		iter2 = lattice_.find(bin);
		//If neighbor exists
		if(iter2!=lattice_.end())
		  {
		    L1CaloCluster neighbor = iter2->second;
		    
		    pattern << "OF:NW Cluster: "<< neighbor.towerE(0) <<
		      " "<< neighbor.towerE(1) << " " << neighbor.towerE(2) <<" "<<
		      neighbor.towerE(3) <<"\n";
		    
		    
		    //Compare the energies and prune if the neighbor has higher Et
		    if(originE < neighbor.E())
		      {
			origin.setTower(0,0);
			central=false;
			
		      }
		  }
		else
		  {
		    pattern << "OF:NW Cluster: 0 0 0 0\n";
		  }
		
	 
		//up
		bin = s.getBin(bin_eta,bin_phi-1);
		iter2 = lattice_.find(bin);
		//If neighbor exists
		if(iter2!=lattice_.end())
		  {
		    L1CaloCluster neighbor = iter2->second;
		    
		    pattern << "OF:N Cluster: "<< neighbor.towerE(0) <<
		      " "<< neighbor.towerE(1) << " " << neighbor.towerE(2) <<" "<<
		      neighbor.towerE(3) <<"\n";
		    
		    
		    //Compare the energies and prune if the neighbor has higher Et
		    if(originE < neighbor.E())
		      {
			origin.setTower(0,0);
			origin.setTower(1,0);
			central=false;
			
		      }
		  }
		else
		  {
		    pattern << "OF:N Cluster: 0 0 0 0\n";
		  }
		
		
		
		//up-right
		bin = s.getBin(bin_eta+1,bin_phi-1);
		iter2 = lattice_.find(bin);
		//If neighbor exists
		if(iter2!=lattice_.end())
		  {
		    L1CaloCluster neighbor = iter2->second;
		    pattern << "OF:NE Cluster: "<< neighbor.towerE(0) <<
		      " "<< neighbor.towerE(1) << " " << neighbor.towerE(2) <<" "<<
		      neighbor.towerE(3) <<"\n";
		    
		    //Compare the energies and prune if the neighbor has higher Et
		    if(originE< neighbor.E())
		      {
			origin.setTower(1,0);
			central=false;
		      }
		  }
		else
		  {
		    pattern << "OF:NE Cluster: 0 0 0 0\n";
		  }
		
		//Check if the cluster is over threshold
		if(origin.E() >=s.clusterThr())
		  {
		    origin.setCentral(central);
		    
		    pattern << "OF:Trimmed Cluster: "<< origin.towerE(0) <<
		      " "<< origin.towerE(1) << " " << origin.towerE(2) <<" "<<
		      origin.towerE(3) <<" Central: "<<central<<" Weights:"<<
		      posIn.first<<" "<<posIn.second;
		    
		    totalET+=origin.E();
		    cleanClusters.push_back(origin);
		    
		  }
		else
		  {
		    pattern << "OF:Trimmed Cluster: 0 0 0 0 Central: 0 Weights: 0 0\n";
		    
		  }
		edm::LogInfo("Overlap Filter Patterns") <<  pattern.str() <<endl;
	      }
	    
	  }

}

std::pair<int,int>
CaloClusterFilteringCard::calculateClusterPosition(l1slhc::L1CaloCluster& cluster)
{
  int etaBit=0;
  int phiBit=0;


  //get et
    double et = ((double)cluster.E())/2.0;
  //    double et = (double)(cluster.E()/2.0);
  
  TriggerTowerGeometry geo;
  
  double eta=0;
  double phi=0;
  double etaW=0;
  double phiW=0;
  
  //eta sum;
  etaW -= cluster.towerE(0);
  etaW -= cluster.towerE(2);
  etaW += cluster.towerE(1);
  etaW += cluster.towerE(3);
  
  etaW= (etaW/cluster.E())+1;
  
  
  if(etaW<0.5)
    {
      eta=geo.eta(cluster.iEta())+geo.towerEtaSize(cluster.iEta())/8;
      etaBit=0;
    }
  else if(etaW<1.0)
    {
      eta=geo.eta(cluster.iEta())+3*geo.towerEtaSize(cluster.iEta())/8;
      etaBit=1;
    }
  else if(etaW<1.5)
    {
      eta=geo.eta(cluster.iEta())+5*geo.towerEtaSize(cluster.iEta())/8;
      etaBit=2;
    }
  else if(etaW<2.0)
    {
      eta=geo.eta(cluster.iEta())+7*geo.towerEtaSize(cluster.iEta())/8;
      etaBit=3;
    }
  
  phiW -= cluster.towerE(0);
  phiW -= cluster.towerE(1);
  phiW += cluster.towerE(2);
  phiW += cluster.towerE(3);
  
  phiW= (phiW/cluster.E())+1;
  
  if(phiW<0.5)
      {
	phi=geo.phi(cluster.iPhi())+geo.towerPhiSize(cluster.iPhi())/8;
	phiBit=0;
      }
  else if(phiW<1.0)
    {
      phi=geo.phi(cluster.iPhi())+3*geo.towerPhiSize(cluster.iPhi())/8;
      phiBit=1;
      }
  else if(phiW<1.5)
    {
      phi=geo.phi(cluster.iPhi())+5*geo.towerPhiSize(cluster.iPhi())/8;
      phiBit=2;
    }
  else if(phiW<2.0)
    {
      phi=geo.phi(cluster.iPhi())+7*geo.towerPhiSize(cluster.iPhi())/8;
      phiBit=3;
    }
  
    std::pair <int,int> p = std::make_pair(etaBit,phiBit);
    math::PtEtaPhiMLorentzVector v(et,eta,phi,0.);
    cluster.setPosBits(etaBit,phiBit);
    cluster.setLorentzVector(v);

  return p;
}





