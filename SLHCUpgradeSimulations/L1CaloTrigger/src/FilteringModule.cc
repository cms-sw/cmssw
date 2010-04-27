#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/FilteringModule.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <sstream>


using namespace l1slhc;
using namespace std;


FilteringModule::FilteringModule():
  L1CaloClusterModule()
{

}

FilteringModule::FilteringModule(const L1CaloTriggerSetup& rcd):
  L1CaloClusterModule(rcd)
{
}


FilteringModule::~FilteringModule()
{

}

void
FilteringModule::populateLattice(const edm::Handle<l1slhc::L1CaloClusterCollection>& clusters)
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
FilteringModule::cleanClusters(const edm::Handle<l1slhc::L1CaloClusterCollection>& clusters,l1slhc::L1CaloClusterCollection& cleanClusters)
{
  //I am doing it the slow way because I want to simulate the Circuit!!

  //Reset and Populate the lattice
  reset();
  populateLattice(clusters);

  //Loop on the Lattice
      for(int bin_phi=s.phiMin();bin_phi<=s.phiMax();++bin_phi)
	for(int bin_eta=0;bin_eta<=s.etaMax()-1;++bin_eta)
          {
	    //Look if there is a cluster here

	    int binOrigin = s.getBin(bin_eta,bin_phi);
	    if(isValid(binOrigin))
	      {
		
		L1CaloCluster origin = *(objectAt(binOrigin));
		double originE = origin.E();
		//Set central bit
		bool central=true;
		int bin=-1;

		//right
		bin = s.getBin(bin_eta+1,bin_phi);
		//If neighbor exists
		if(isValid(bin))
		  {
		    L1CaloClusterRef neighbor = objectAt(bin);
		    
		    //Compare the energies and prune if the neighbor has higher Et
		    if(originE <= neighbor->E())
		      {
			
			origin.removeConstituent(1,0);
			origin.removeConstituent(1,1);
			central=false;
		      }
		  }
		//right-down
		bin = s.getBin(bin_eta+1,bin_phi+1);
		//If neighbor exists
		if(isValid(bin))
		  {
		    L1CaloClusterRef neighbor = objectAt(bin);
		    
		    //Compare the energies and prune if the neighbor has higher Et
		    if(originE <= neighbor->E())
		      {
			origin.removeConstituent(1,1);
			central=false;
		      }
		  }

		//down
		bin = s.getBin(bin_eta,bin_phi+1);
		//If neighbor exists
		if(isValid(bin))
		  {
		    L1CaloClusterRef neighbor = objectAt(bin);
		    
		    //Compare the energies and prune if the neighbor has higher Et
		    if(originE <= neighbor->E())
		      {
			origin.removeConstituent(0,1);
			origin.removeConstituent(1,1);
			central=false;
		      }
		  }

		//down-left
		bin = s.getBin(bin_eta-1,bin_phi+1);
		//If neighbor exists
		if(isValid(bin))
		  {
		    L1CaloClusterRef neighbor = objectAt(bin);
		    
		    //Compare the energies and prune if the neighbor has higher Et
		    if(originE <= neighbor->E())
		      {
			origin.removeConstituent(0,1);
			central=false;
		      }
		  }

		//left
		bin = s.getBin(bin_eta-1,bin_phi);
		//If neighbor exists
		if(isValid(bin))
		  {
		    L1CaloClusterRef neighbor = objectAt(bin);
		    
		    //Compare the energies and prune if the neighbor has higher Et
		    if(originE < neighbor->E())
		      {
			origin.removeConstituent(0,0);
			origin.removeConstituent(0,1);
			central=false;
		      }
		  }

		//left-up
		bin = s.getBin(bin_eta-1,bin_phi-1);
		//If neighbor exists
		if(isValid(bin))
		  {
		    L1CaloClusterRef neighbor = objectAt(bin);
		    
		    //Compare the energies and prune if the neighbor has higher Et
		    if(originE < neighbor->E())
		      {
			origin.removeConstituent(0,0);
			central=false;
		      }
		  }

		//up
		bin = s.getBin(bin_eta,bin_phi-1);
		//If neighbor exists
		if(isValid(bin))
		  {
		    L1CaloClusterRef neighbor = objectAt(bin);
		    
		    //Compare the energies and prune if the neighbor has higher Et
		    if(originE < neighbor->E())
		      {
			origin.removeConstituent(0,0);
			origin.removeConstituent(1,0);
			central=false;
		      }
		  }

		//up-right
		bin = s.getBin(bin_eta+1,bin_phi-1);
		//If neighbor exists
		if(isValid(bin))
		  {
		    L1CaloClusterRef neighbor = objectAt(bin);
		    
		    //Compare the energies and prune if the neighbor has higher Et
		    if(originE < neighbor->E())
		      {

			origin.removeConstituent(1,0);
			central=false;
		      }
		  }



		
		//Check if the cluster is over threshold
		if(origin.E() >=s.clusterThr())
		  {
		    std::pair<int,int> posIn =  calculateClusterPosition(origin);
		    origin.setCentral(central);
		    cleanClusters.push_back(origin);
		  }
	      }
	  }
}


std::pair<int,int>
FilteringModule::calculateClusterPosition(l1slhc::L1CaloCluster& cluster)
{
  int etaBit=0;
  int phiBit=0;


  //get et
  double et =(double) (cluster.E()/2);

  TriggerTowerGeometry geo;

  double eta=0;
  double phi=0;
  double etaW=0;
  double phiW=0;

  int pos=-1;

  //eta sum;
  pos=cluster.hasConstituent(0,0);
  if(pos!=-1) 
    etaW-=cluster.getConstituent(pos)->E()+cluster.getConstituent(pos)->H();
  pos=cluster.hasConstituent(0,1);
  if(pos!=-1) 
    etaW-=cluster.getConstituent(pos)->E()+cluster.getConstituent(pos)->H();
  pos=cluster.hasConstituent(1,0);
  if(pos!=-1) 
    etaW+=cluster.getConstituent(pos)->E()+cluster.getConstituent(pos)->H();
  pos=cluster.hasConstituent(1,1);
  if(pos!=-1) 
    etaW+=cluster.getConstituent(pos)->E()+cluster.getConstituent(pos)->H();
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
  

  pos=cluster.hasConstituent(0,0);
  if(pos!=-1) 
    phiW-=cluster.getConstituent(pos)->E()+cluster.getConstituent(pos)->H();

  pos=cluster.hasConstituent(1,0);
  if(pos!=-1) 
    phiW-=cluster.getConstituent(pos)->E()+cluster.getConstituent(pos)->H();

  pos=cluster.hasConstituent(0,1);
  if(pos!=-1) 
    phiW+=cluster.getConstituent(pos)->E()+cluster.getConstituent(pos)->H();

  pos=cluster.hasConstituent(1,1);
  if(pos!=-1) 
    phi+=cluster.getConstituent(pos)->E()+cluster.getConstituent(pos)->H();
    
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




