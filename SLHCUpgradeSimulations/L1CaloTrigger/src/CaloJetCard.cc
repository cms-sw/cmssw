#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/CaloJetCard.h"
using namespace l1slhc;
using namespace std;


CaloJetCard::CaloJetCard()
{

}

CaloJetCard::CaloJetCard(const L1CaloTriggerSetup& rcd)
{
  s = rcd;
}


CaloJetCard::~CaloJetCard()
{

}


void
CaloJetCard::populateLattice(const l1slhc::L1CaloClusterCollection& clusters)
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
CaloJetCard::makeJets(const l1slhc::L1CaloClusterCollection& clusters,l1slhc::L1CaloJetCollection& jets)
{


  //Reset and Populate the lattice
  reset();
  populateLattice(clusters);

  //Loop on the Lattice
      for(int bin_phi=s.phiMin();bin_phi<=s.phiMax();++bin_phi)
  for(int bin_eta=0;bin_eta<=s.etaMax()-1;++bin_eta)
          {
           //Look if there is a cluster here and if the cluster is central (not pruned)
      //Those central maxima will be the builders of the Jet
         std::map<int,L1CaloCluster>::iterator iter;
         iter = lattice_.find(s.getBin(bin_eta,bin_phi));
      if(iter!=lattice_.end())
        if(iter->second.isCentral())
       {

           L1CaloCluster origin = iter->second;

         //Jet Sums
         int ET=0;
         int ETETA=0;
         int ETPHI=0;

         //There is a cluster here:Sum energy
         for(int phi=bin_phi-3;phi<=bin_phi+3+1;++phi)
           for(int eta=bin_eta-3;eta<=bin_eta+3+1;++eta)
         {
           //Take this cluster
           int bin = s.getBin(eta,phi);
           std::map<int,L1CaloCluster>::iterator iter2= lattice_.find(bin);
           if(iter2!=lattice_.end())
             {

               ET+=iter2->second.E();
               ETETA+=eta*iter2->second.E();
               ETPHI+=phi*iter2->second.E();

             }

         }

         //OK Now Calculate Cut OFfs
         if(abs(ETETA/ET-bin_eta)<s.jetCenterDev() && abs(ETPHI/ET-bin_phi)<s.jetCenterDev() && ET>s.minJetET())
           jets.push_back(L1CaloJet(origin.iEta(),origin.iPhi(),ET));

       }
    }

      filterJets(jets);
}






void
CaloJetCard::filterJets(l1slhc::L1CaloJetCollection& jets)
{
  L1CaloJetCollection filteredJets;

  if(jets.size()>0)
    {

      //Sort them
      std::sort(jets.begin(),jets.end(),HigherJetEt());
      filteredJets.push_back(jets[0]);

      for(size_t i=1;i<jets.size();++i)
  {
    bool add=true;
    for(size_t j=1;j<filteredJets.size();++j)
      {
        if(abs(jets[i].iEta()-filteredJets[j].iEta())<4 && abs(jets[i].iPhi()-filteredJets[j].iPhi())<4)
    add =false;

      }
    if(add)
      filteredJets.push_back(jets[i]);
  }


      jets = filteredJets;
    }

}






