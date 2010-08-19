#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/JetModule.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <sstream>


using namespace l1slhc;
using namespace std;

JetModule::JetModule():
  L1CaloRegionModule()
{


}

JetModule::JetModule(const L1CaloTriggerSetup& rcd):
  L1CaloRegionModule(rcd)
{

}
JetModule::~JetModule()
{
}


void
JetModule::populateLattice(const edm::Handle<L1CaloRegionCollection>& regions)
{
  //Loop On the Hits and see which Hits are inside the region. Then Populate the region



  if(regions->size()>0)
      for(L1CaloRegionCollection::const_iterator Region = regions->begin();Region!=regions->end();++Region)
        for(std::map<int,std::pair< int,int > >::iterator it = s.geoMap_.begin();it!=s.geoMap_.end();++it)
            if(it->second.first==Region->iEta() && it->second.second==Region->iPhi())
	      {
		L1CaloRegionRef ref(regions,Region-regions->begin());
		lattice_[it->first] = ref;
	      }

}

void
JetModule::clusterize(L1CaloJetCollection& jets,const edm::Handle<L1CaloRegionCollection>& regions)
{
  //I am doing it the slow way because I want to simulate the Circuit!!
  reset();

  //Populate the lattice with ECAL
  populateLattice(regions);

  //Perform the Sliding Window Algorithm
  //The last row and  column are repeated  for the overlap
  for(int phi =s.phiMin() ;phi<=s.phiMax()-7;phi+=4)
    for(int eta =s.etaMin() ;eta<=s.etaMax()-7;eta+=4)
    {
      int E=0;
      int binOrigin = s.getBin(eta,phi);
      std::pair<int,int> p  =s.geoMap_[binOrigin];
      

      L1CaloJet jet(p.first,p.second);

      for(int bin_phi=phi;bin_phi<=phi+4;bin_phi+=4)
	for(int bin_eta=eta;bin_eta<=eta+4;bin_eta+=4)
	  {
	    int bin = s.getBin(bin_eta,bin_phi);
	    if(isValid(bin))
	      {
		jet.addConstituent(objectAt(bin));
	      }
	  }
      if(jet.E()>0)
	{
	  calculateJetPosition(jet);
	  jets.push_back(jet);
	}
    }

}



void 
JetModule::calculateJetPosition(l1slhc::L1CaloJet& jet)
{
  int et = jet.E();


  //Calculate float value of eta for barrel+endcap(L.Gray)
  double eta =-1982.;//an important year...
  double etaOffset=0.087/2.0;
  int abs_eta = abs(jet.iEta()+4);
  const double endcapEta[8] = {0.09,0.1,0.113,0.129,0.15,0.178,0.15,0.35};
  if(abs_eta <=20)
    {
      eta =  (abs_eta*0.0870)-etaOffset;
    }
  else
    {
      int offset = abs(jet.iEta()+4) -21;
      eta = (20*0.0870);//-etaOffset;
      for(int i = 0;i<= offset;++i)
	{
	  eta+=endcapEta[i];
	}
      eta-=endcapEta[abs(jet.iEta()+4)-21]/2.;
    }

  if(jet.iEta()+4<0) eta  = -eta;
  
  double phi = ((jet.iPhi()+4)*0.087)-0.087/2.;
  double Et= ((double)et)/2.;



 math::PtEtaPhiMLorentzVector p(Et,eta,phi,0. ) ;
 jet.setP4(p);

}
