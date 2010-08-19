#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/JetFilteringModule.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <sstream>


using namespace l1slhc;
using namespace std;


JetFilteringModule::JetFilteringModule():
  L1CaloJetModule()
{

}

JetFilteringModule::JetFilteringModule(const L1CaloTriggerSetup& rcd):
  L1CaloJetModule(rcd)
{
}


JetFilteringModule::~JetFilteringModule()
{

}

void
JetFilteringModule::populateLattice(const edm::Handle<l1slhc::L1CaloJetCollection>& clusters)
{
  if(clusters->size()>0)
    for(l1slhc::L1CaloJetCollection::const_iterator cl = clusters->begin();cl!=clusters->end();++cl)
      for(std::map<int,std::pair< int,int > >::iterator it = s.geoMap_.begin();it!=s.geoMap_.end();++it)
	if(it->second.first==cl->iEta() && it->second.second==cl->iPhi())
          {
	    L1CaloJetRef ref(clusters,cl-clusters->begin());
	    lattice_[it->first] = ref;
          }
}


void
JetFilteringModule::cleanClusters(const edm::Handle<l1slhc::L1CaloJetCollection>& clusters,l1slhc::L1CaloJetCollection& cleanClusters)
{
  //I am doing it the slow way because I want to simulate the Circuit!!

  //Reset and Populate the lattice
  reset();
  populateLattice(clusters);

  //Loop on the Lattice
      for(int bin_phi=s.phiMin();bin_phi<=s.phiMax();bin_phi+=4)
	for(int bin_eta=0;bin_eta<=s.etaMax()-3;bin_eta+=4)
          {
	    //Look if there is a cluster here
	    int binOrigin = s.getBin(bin_eta,bin_phi);
	    if(isValid(binOrigin))
	      {
		
		L1CaloJet origin = *(objectAt(binOrigin));
		double originE = origin.E();

		//Set central bit
		bool central=true;
		int bin=-1;

		//right
		bin = s.getBin(bin_eta+4,bin_phi);
		//If neighbor exists
		if(isValid(bin))
		  {
		    L1CaloJetRef neighbor = objectAt(bin);
		    
		    //Compare the energies and prune if the neighbor has higher Et
		    if(originE <= neighbor->E())
		      {
			origin.removeConstituent(4,0);
			origin.removeConstituent(4,4);
			central=false;
		      }
		  }

		//right-down
		bin = s.getBin(bin_eta+4,bin_phi+4);
		//If neighbor exists
		if(isValid(bin))
		  {
		    L1CaloJetRef neighbor = objectAt(bin);
		    
		    //Compare the energies and prune if the neighbor has higher Et
		    if(originE <= neighbor->E())
		      {
			origin.removeConstituent(4,4);
			central=false;
		      }
		  }

		//down
		bin = s.getBin(bin_eta,bin_phi+4);
		//If neighbor exists
		if(isValid(bin))
		  {
		    L1CaloJetRef neighbor = objectAt(bin);
		    
		    //Compare the energies and prune if the neighbor has higher Et
		    if(originE <= neighbor->E())
		      {

			origin.removeConstituent(0,4);
			origin.removeConstituent(4,4);
			central=false;
		      }
		  }

		//down-left
		bin = s.getBin(bin_eta-4,bin_phi+4);
		//If neighbor exists
		if(isValid(bin))
		  {
		    L1CaloJetRef neighbor = objectAt(bin);
		    
		    //Compare the energies and prune if the neighbor has higher Et
		    if(originE <= neighbor->E())
		      {
			origin.removeConstituent(0,4);
			central=false;
		      }
		  }

		//left
		bin = s.getBin(bin_eta-4,bin_phi);
		//If neighbor exists
		if(isValid(bin))
		  {
		    L1CaloJetRef neighbor = objectAt(bin);
		    
		    //Compare the energies and prune if the neighbor has higher Et
		    if(originE < neighbor->E())
		      {
			origin.removeConstituent(0,0);
			origin.removeConstituent(0,4);
			central=false;
		      }
		  }

		//left-up
		bin = s.getBin(bin_eta-4,bin_phi-4);
		//If neighbor exists
		if(isValid(bin))
		  {
		    L1CaloJetRef neighbor = objectAt(bin);
		    
		    //Compare the energies and prune if the neighbor has higher Et
		    if(originE < neighbor->E())
		      {
			origin.removeConstituent(bin_eta,bin_phi);
			central=false;
		      }
		  }

		//up
		bin = s.getBin(bin_eta,bin_phi-4);
		//If neighbor exists
		if(isValid(bin))
		  {
		    L1CaloJetRef neighbor = objectAt(bin);
		    
		    //Compare the energies and prune if the neighbor has higher Et
		    if(originE < neighbor->E())
		      {
			origin.removeConstituent(0,0);
			origin.removeConstituent(4,0);
			central=false;
		      }
		  }

		//up-right
		bin = s.getBin(bin_eta+4,bin_phi-4);
		//If neighbor exists
		if(isValid(bin))
		  {
		    L1CaloJetRef neighbor = objectAt(bin);
		    
		    //Compare the energies and prune if the neighbor has higher Et
		    if(originE < neighbor->E())
		      {
			origin.removeConstituent(4,0);
			central=false;
		      }
		  }


		//Check if the jet is over threshold
		if(origin.E() >=s.minJetET())
		  {
		    origin.setCentral(central);
		    cleanClusters.push_back(origin);
		  }
	      }
	  }
}


