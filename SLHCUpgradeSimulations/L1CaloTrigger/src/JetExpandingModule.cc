#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/JetExpandingModule.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <sstream>


using namespace l1slhc;
using namespace std;


JetExpandingModule::JetExpandingModule():
  L1CaloJetModule()
{

}

JetExpandingModule::JetExpandingModule(const L1CaloTriggerSetup& rcd):
  L1CaloJetModule(rcd)
{
}


JetExpandingModule::~JetExpandingModule()
{

}

void
JetExpandingModule::populateLattice(const edm::Handle<l1slhc::L1CaloJetCollection>& clusters)
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
JetExpandingModule::expandClusters(const edm::Handle<l1slhc::L1CaloJetCollection>& clusters,l1slhc::L1CaloJetCollection& cleanClusters)
{
  //I am doing it the slow way because I want to simulate the Circuit!!

  //Reset and Populate the lattice
  reset();
  populateLattice(clusters);
  addersInput.clear();

  //STEP 1 : Preloop on the lattice and fill the adders input

      for(int bin_phi=s.phiMin();bin_phi<=s.phiMax();bin_phi+=4)
	for(int bin_eta=0;bin_eta<=s.etaMax()-3;bin_eta+=4)
          {
	    //Look if there is a non central cluster here

	    int binOrigin = s.getBin(bin_eta,bin_phi);
	    if(isValid(binOrigin))
	      {
		
		L1CaloJet origin = *(objectAt(binOrigin));
		if(!origin.central()) 
		  {
		    //loop and find the highest cebtral neighbor
		    int highestCentralNeigborBin=-1000;
		    int highestCentralNeigborE=-1000;
		    for(int i=-4;i<=4;i+=4)		    
		      for( int j=-4;j<=4;j+=4)
			if(!(i==0&&j==0))
			  {
			    int bin = s.getBin(bin_eta+i,bin_phi+j);
			    if(isValid(bin))
			      if(objectAt(bin)->central())
				{
				  if(objectAt(bin)->E()>highestCentralNeigborE)
				    {
				      highestCentralNeigborE=objectAt(bin)->E();
				      highestCentralNeigborBin=bin;
				    }
				}
			  }

		    if(isValid(highestCentralNeigborBin))
		      {
			//if entries exist addd also the constituents of this cluster
			
			if(addersInput.find(highestCentralNeigborBin)!=addersInput.end()) {
			  L1CaloRegionRefVector vec=addersInput[highestCentralNeigborBin];
			  if(origin.getConstituents().size()>0){

			    for(unsigned int k=0;k<origin.getConstituents().size();++k)
			      {

				vec.push_back(origin.getConstituents().at(k));
			      }

			  }

			  addersInput[highestCentralNeigborBin] = vec;
			  
			}
			else
			  {
				  
			    L1CaloRegionRefVector vec=origin.getConstituents();
			    addersInput[highestCentralNeigborBin] = vec;
			  }
		      }
		  }
	      }
	  }
      
      ///OK Now apply the adders to create the composite jet candidates
      for(int bin_phi=s.phiMin();bin_phi<=s.phiMax();bin_phi+=4)
	for(int bin_eta=s.etaMin();bin_eta<=s.etaMax()-3;bin_eta+=4)
          {
	    //Look if there is a non central cluster here
	    int binOrigin = s.getBin(bin_eta,bin_phi);
	    if(isValid(binOrigin))
	      {
		L1CaloJet origin = *(objectAt(binOrigin));
		if(origin.central()) 
		  {
      
		    //get The adders input and include the neighbors
		    if(addersInput.find(binOrigin)!=addersInput.end())
		      {
			L1CaloRegionRefVector sums = addersInput[binOrigin];
			for(unsigned int i=0;i<sums.size();++i)
			  origin.addConstituent(sums.at(i));
		      }
		    //change the ET!
		    origin.setP4(math::PtEtaPhiMLorentzVector(((double)origin.E())/2.,origin.p4().eta(),origin.p4().phi(),0.0));
		    cleanClusters.push_back(origin);
		  }
	      }
	  }
}
