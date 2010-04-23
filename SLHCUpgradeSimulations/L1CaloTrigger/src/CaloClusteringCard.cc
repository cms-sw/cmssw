#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/CaloClusteringCard.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <sstream>


using namespace l1slhc;
using namespace std;

CaloClusteringCard::CaloClusteringCard()
{


}

CaloClusteringCard::CaloClusteringCard(const L1CaloTriggerSetup& rcd)
{
  s = rcd;
}
CaloClusteringCard::~CaloClusteringCard()
{
}


void
CaloClusteringCard::populateLattice(const L1CaloTowerCollection& towers)
{
  //Loop On the Hits and see which Hits are inside the region. Then Populate the region


  //Do it for ECAL
  if(towers.size()>0)
      for(L1CaloTowerCollection::const_iterator Tower = towers.begin();Tower!=towers.end();++Tower)
        for(std::map<int,std::pair< int,int > >::iterator it = s.geoMap_.begin();it!=s.geoMap_.end();++it)
            if(it->second.first==Tower->iEta() && it->second.second==Tower->iPhi())
	      {
		lattice_[it->first] = *Tower;
	      }

}

void
CaloClusteringCard::clusterize(L1CaloClusterCollection& clusters,const L1CaloTowerCollection& towers)
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
      //Also caluclate the total ET for Cross check!
      
      //Get lattice Point
      int bin = s.getBin(eta,phi);
      //Create a cluster
      std::pair<int,int> p  =s.geoMap_[bin];
      L1CaloCluster cl(p.first,p.second);
      
      bool fineGrain=false;
      int ECALE=0;
      int E=0;
      int bitmask=0;
      
      //(0,0)
      //Create pattern specific variables
      std::ostringstream pattern;
      bool printPattern=false;
      
      
      bin = s.getBin(eta,phi);
      if(lattice_.find(bin)!=lattice_.end() )
	{
	  pattern << lattice_[bin].E() << " " << lattice_[bin].H() << " " << lattice_[bin].fineGrain();
	  
	  //Thresholding
	  if(lattice_[bin].E() >=s.ecalActivityThr() || lattice_[bin].H()>=s.hcalActivityThr())
	    {
	      printPattern =true;
	      int et = lattice_[bin].E()+lattice_[bin].H();
	      cl.setTower(0,et);//Add the tower energy
	      E+=et;
	      ECALE+=lattice_[bin].E();
	      bitmask+=1;
	      fineGrain = fineGrain||lattice_[bin].fineGrain();
	    }
	}
      else
	pattern << "0 0 0";
      
      
      //(1,0)
      bin = s.getBin(eta+1,phi);
      if(lattice_.find(bin)!=lattice_.end() )
	{
	  pattern << " " << lattice_[bin].E() << " " << lattice_[bin].H() << " " << lattice_[bin].fineGrain();
	  
	  //Thresholding
	 if(lattice_[bin].E() >=s.ecalActivityThr() || lattice_[bin].H()>=s.hcalActivityThr())
	   {
	     printPattern =true;
	     int et = lattice_[bin].E()+lattice_[bin].H();
	     cl.setTower(1,et);//Add the tower energy
	     E+=et;
	     ECALE+=lattice_[bin].E();
	     bitmask+=2;
	     fineGrain = fineGrain||lattice_[bin].fineGrain();
	     
	   }

       }
     else
       pattern << " 0 0 0";

     //(0,1)
     bin = s.getBin(eta,phi+1);
     if(lattice_.find(bin)!=lattice_.end() )
       {
	 pattern << " " << lattice_[bin].E() << " " << lattice_[bin].H() << " " << lattice_[bin].fineGrain();
	 
	 //Thresholding
	 if(lattice_[bin].E() >=s.ecalActivityThr() || lattice_[bin].H()>=s.hcalActivityThr())
	   {
	     int et = lattice_[bin].E()+lattice_[bin].H();
	     cl.setTower(2,et);//Add the tower energy
	     E+=et;
	     ECALE+=lattice_[bin].E();
	     bitmask+=4;
	     fineGrain = fineGrain||lattice_[bin].fineGrain();
	   }
       }
     else
       pattern << " 0 0 0";
     
     
     //(1,1)
     bin = s.getBin(eta+1,phi+1);
     if(lattice_.find(bin)!=lattice_.end() )
       {
	 pattern << " " << lattice_[bin].E() << " " << lattice_[bin].H() << " " << lattice_[bin].fineGrain();
	 
	 //Thresholding
	 if(lattice_[bin].E() >=s.ecalActivityThr() || lattice_[bin].H()>=s.hcalActivityThr())
	   {
	     printPattern =true;
	     int et = lattice_[bin].E()+lattice_[bin].H();
	     cl.setTower(3,et);//Add the tower energy
	     E+=et;
	     ECALE+=lattice_[bin].E();
	     bitmask+=8;
	     fineGrain = fineGrain||lattice_[bin].fineGrain();
	   }
       }
     else
       pattern << " 0 0 0";

     
     
     //Calculate Electron Cut
     int electronValue =(int)(100.*((double)ECALE)/((double)E));
     

     //Calculate Tau Lead Tower Cut
     cl.setLeadTowerTau( cl.seedTowerE() >= s.seedTowerThr());
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
     
     pattern << " " << cl.towerE(0) << " " << cl.towerE(1) << " "<< cl.towerE(2) << " " << cl.towerE(3) << " " << cl.eGamma() << " " << cl.fg();
     clusters.push_back(cl);

     //Output Pattern
     if(printPattern)
       {
	 edm::LogInfo("ClusteringPatterns") << "CLUSTERING_PATTERN: " << pattern.str() <<endl;
       }
    }
}



