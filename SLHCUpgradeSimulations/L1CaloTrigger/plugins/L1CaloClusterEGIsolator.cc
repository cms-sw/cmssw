//#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/L1CaloAlgoBase.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/SLHC/interface/L1CaloTriggerSetup.h"
#include "SimDataFormats/SLHC/interface/L1CaloTriggerSetupRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "SimDataFormats/SLHC/interface/L1CaloCluster.h"
#include "SimDataFormats/SLHC/interface/L1CaloClusterFwd.h"
#include "SimDataFormats/SLHC/interface/L1CaloTowerFwd.h"
#include "SimDataFormats/SLHC/interface/L1CaloTower.h"

#include "SimDataFormats/SLHC/interface/L1TowerNav.h"

//author: Sam Harper (RAL, 22/05/2013)
//WARNING: this class is not yet complete, its still in development phase
//all it does right now is calculate the isolation for EG

class L1CaloClusterEGIsolator : public edm::EDProducer {
private:
  edm::InputTag caloClustersTag_;
  edm::InputTag caloTowersTag_;
  edm::InputTag rhoTag_;

  edm::Handle<l1slhc::L1CaloClusterCollection> caloClustersHandle_;
  edm::Handle<l1slhc::L1CaloTowerCollection> caloTowersHandle_;
  edm::Handle<int> rhoHandle_;
  
  edm::ESHandle <L1CaloTriggerSetup> caloTriggerSetup_;
  

  int phiOffset_;
  int etaOffset_;
  int phiIncrement_;
  int etaIncrement_;

  int isolEtCut_; // cut is <= isolEtCut

  int maxTowerIEta_; //max ieta of towers to include in sum (ie includes iEta <=maxTowerIEta_)
public:
  L1CaloClusterEGIsolator(const edm::ParameterSet & );
  ~L1CaloClusterEGIsolator();
  
  void algorithm(const int &, const int &,l1slhc::L1CaloClusterCollection&);
  //  std::string sourceName() const;
  void produce(edm::Event &, const edm::EventSetup &);
  

  int egEcalIsolation(int iEta,int iPhi)const; //takes reco style iPhi, iEta, ie -32, 32, 1 to 72
  int egHcalIsolation(int iEta,int iPhi)const;//takes reco style iPhi, iEta
  
  template<class T> typename T::const_iterator getObject(const int& trigEta,const int &trigPhi,const edm::Handle<T>& handle)const;
  
  
  
};


L1CaloClusterEGIsolator::L1CaloClusterEGIsolator(const edm::ParameterSet & config):
  phiOffset_(0),
  etaOffset_(-1),
  phiIncrement_(1),
  etaIncrement_(1)
{
  caloClustersTag_ = config.getParameter<edm::InputTag>("caloClustersTag");
  caloTowersTag_ = config.getParameter<edm::InputTag>("caloTowersTag");
  rhoTag_ = config.getParameter<edm::InputTag>("rhoTag");
  maxTowerIEta_ = config.getParameter<int>("maxTowerIEta");
  isolEtCut_ = config.getParameter<int>("isolEtCut");
  

  produces<l1slhc::L1CaloClusterCollection>();
}


L1CaloClusterEGIsolator::~L1CaloClusterEGIsolator(  )
{
}

void L1CaloClusterEGIsolator::produce(edm::Event & iEvent, const edm::EventSetup & iSetup)
{
  iSetup.get<L1CaloTriggerSetupRcd>().get(caloTriggerSetup_);
   
  iEvent.getByLabel(caloClustersTag_,caloClustersHandle_);
  iEvent.getByLabel(caloTowersTag_,caloTowersHandle_);
  iEvent.getByLabel(rhoTag_,rhoHandle_);
 
  int trigPhiMin = caloTriggerSetup_->phiMin();
  int trigPhiMax = caloTriggerSetup_->phiMax() + phiOffset_;
  int trigEtaMin = caloTriggerSetup_->etaMin();
  int trigEtaMax = caloTriggerSetup_->etaMax() + etaOffset_;
  
  std::auto_ptr<l1slhc::L1CaloClusterCollection> outputCollection(new l1slhc::L1CaloClusterCollection);
  for(int trigEtaNr=trigEtaMin; trigEtaNr<=trigEtaMax; trigEtaNr+=etaIncrement_ ){
    for (int trigPhiNr=trigPhiMin; trigPhiNr<=trigPhiMax; trigPhiNr+=phiIncrement_ ){
      this->algorithm(trigEtaNr,trigPhiNr,*outputCollection);
    }
  }
  iEvent.put(outputCollection);
}

//so instead of looping over the collection, it checks all possible eta/phi values for the presence of a clusters
//somewhat strange, sure theres a reason, probably as its more paralisable to better for hardware, so keep for now
void L1CaloClusterEGIsolator::algorithm( const int &trigEta, const int &trigPhi,l1slhc::L1CaloClusterCollection& outputColl)
{

  //  std::cout <<"aEta "<<aEta <<" aPhi "<<aPhi<<std::endl;
  l1slhc::L1CaloClusterCollection::const_iterator clusIt = getObject(trigEta,trigPhi,caloClustersHandle_);
  if(clusIt!=caloClustersHandle_->end()) { 

    if(clusIt->isCentral()){
      l1slhc::L1CaloCluster newCluster(*clusIt);
      //we may want to cut on em + had seperately, right now we dont as at this WP, the performance is the same but it changes for other efficiencies
      int emIsolEt = egEcalIsolation(newCluster.iEta(),newCluster.iPhi());
      int hadIsolEt = egHcalIsolation(newCluster.iEta(),newCluster.iPhi());
      newCluster.setIsoEmAndHadEtEG(emIsolEt,hadIsolEt);  

      int nrTowers = caloTowersHandle_->size(); //we are using this as a proxy for rho for now. this will change
      int rho = nrTowers/40; //it seems to work, although at 140 PU it may not
      if(emIsolEt+hadIsolEt-rho <=isolEtCut_) newCluster.setIsoEG(true);
      outputColl.insert(newCluster.iEta(),newCluster.iPhi(),newCluster);
    }
    
  }
}



//going to experiment with different vetos for HCAL and ECAL isolation
//this is simple veto, 2 tower wide
//iEta, iPhi is the x of the below diagram
//  |0|0|
//  |x|0|
//this 2x2 is also the highest 2x2 in the local area from the clustering algorithm
//therefore the veto is iEta and iEta + 1 for ecal
int L1CaloClusterEGIsolator::egEcalIsolation(int iEta,int iPhi)const
{
  
  //first we need the lead tower 
  int leadTowerEmEt = 0;
  int leadTowerIEta = iEta;
  int leadTowerIPhi = iPhi;
  for(int etaNr=0;etaNr<=1;etaNr++){
    for(int phiNr=0;phiNr<=1;phiNr++){
      int towerIEta = L1TowerNav::getOffsetIEta(iEta,etaNr);
      int towerIPhi = L1TowerNav::getOffsetIPhi(iEta,iPhi,phiNr);
      l1slhc::L1CaloTowerCollection::const_iterator towerIt = caloTowersHandle_->find(towerIEta,towerIPhi);
      if(towerIt!=caloTowersHandle_->end() && towerIt->E()>leadTowerEmEt){
	leadTowerEmEt = towerIt->E();
	leadTowerIPhi = towerIt->iPhi();
	leadTowerIEta = towerIt->iEta();
      }
    }
  }
  
  int isolEmEt=0;
  //now loop over +/-2 towers in eta and +/-4 in phi (be carefull in the endcap)
  for(int etaNr=-2;etaNr<=2;etaNr++){
    int towerIEta = L1TowerNav::getOffsetIEta(leadTowerIEta,etaNr);
    if(abs(towerIEta)>maxTowerIEta_) continue; //outside allowed eta region (aka likely HF in normal config)

    for(int phiNr=-4;phiNr<=4;phiNr++){
      int towerIPhi = L1TowerNav::getOffsetIPhi(towerIEta,leadTowerIPhi,phiNr);
      
      //vetoing +/-2 in phi from leadTower and iEta = cluster which is towerIEta and towerIEta+1
      if(abs(phiNr)<=2){
	if(towerIEta==iEta) continue;
	if(towerIEta==L1TowerNav::getOffsetIEta(iEta,1)) continue; //eta veto region
      }
      l1slhc::L1CaloTowerCollection::const_iterator towerIt = caloTowersHandle_->find(towerIEta,towerIPhi);
      if(towerIt!=caloTowersHandle_->end()) isolEmEt+=towerIt->E();
    }//end phi loop
  }//end eta loop
  
  return isolEmEt;

}

//going to experiment with different vetos for HCAL and ECAL isolation
//this is simple veto, 2 tower wide
//iEta, iPhi is the x of the below diagram
//  |0|0|
//  |x|0|
//this 2x2 is also the highest 2x2 in the local area from the clustering algorithm
//therefore the veto is iEta and iEta + 1 and iPhi and iPhi+1
int L1CaloClusterEGIsolator::egHcalIsolation(int iEta,int iPhi)const
{
  // if(iEta!=-26 || iPhi!=22) return 0; 

  //first we need the lead tower 
  int leadTowerEmEt = 0;
  int leadTowerIEta = iEta;
  int leadTowerIPhi = iPhi;
  for(int etaNr=0;etaNr<=1;etaNr++){
    for(int phiNr=0;phiNr<=1;phiNr++){
      int towerIEta = L1TowerNav::getOffsetIEta(iEta,etaNr);
      int towerIPhi = L1TowerNav::getOffsetIPhi(towerIEta,iPhi,phiNr);
      l1slhc::L1CaloTowerCollection::const_iterator towerIt = caloTowersHandle_->find(towerIEta,towerIPhi);
      if(towerIt!=caloTowersHandle_->end() && towerIt->E()>leadTowerEmEt){ //therefore in case of ties, its 0,0 first then, 0,1, 1,0 and 1,1, it is unclear if this is optimal behaviour but will do for now
	leadTowerEmEt = towerIt->E();
	leadTowerIPhi = towerIt->iPhi();
	leadTowerIEta = towerIt->iEta();
      }
    }
  }
  
  // std::cout <<"iEta "<<iEta<<" iPhi "<<iPhi<<" lead tower eta "<<leadTowerIEta <<" phi "<<leadTowerIPhi<<std::endl;

  

  int isolHadEt=0;
  //now loop over +/-2 towers in eta and +/- 4 in phi (be carefull in the endcap)
  for(int etaNr=-2;etaNr<=2;etaNr++){
    int towerIEta = L1TowerNav::getOffsetIEta(leadTowerIEta,etaNr);
    if(abs(towerIEta)>maxTowerIEta_) continue; //outside allowed eta region (aka likely HF in normal config)
    for(int phiNr=-4;phiNr<=4;phiNr++){
      int towerIPhi = L1TowerNav::getOffsetIPhi(towerIEta,leadTowerIPhi,phiNr);

      //    std::cout <<"towerIEta "<<towerIEta<<" towerIPhi "<<towerIPhi<<" iphi "<<iPhi<<" offset iphi "<<L1TowerNav::getOffsetIPhi(iEta,iPhi,phiNr)<<std::endl;

      //somewhat confusing, should just be checking if the tower in is in the 1x2 of the cluster with eta = leadTower, its below leadTower for -ve eta, above lead Tower for +ve eta
      int localPhiVeto = iEta>0 ? 1 : -1;
      if(etaNr==0 && (phiNr==0 || phiNr==localPhiVeto)) continue; //eta veto region
      
      l1slhc::L1CaloTowerCollection::const_iterator towerIt = caloTowersHandle_->find(towerIEta,towerIPhi);
   
      if(towerIt!=caloTowersHandle_->end()) isolHadEt+=towerIt->H();
      //std::cout <<"   "<<" hcal et : "<<towerIt->H()<<" tot et "<<isolHadEt<<" ieta "<<towerIt->iEta()<<" iphi "<<towerIt->iPhi()<<std::endl;
      //}else{
      //	std::cout <<"   "<<"not valid"<<std::endl;
      //}
    }//end phi loop
  }//end eta loop
  
  return isolHadEt;

}


template<class T> typename T::const_iterator L1CaloClusterEGIsolator::getObject(const int& trigEta,const int &trigPhi,const edm::Handle<T>& handle)const
{
  int index = caloTriggerSetup_->getBin(trigEta,trigPhi);
  std::pair<int,int> etaPhi = caloTriggerSetup_->getTowerEtaPhi(index);
 
  return handle->find(etaPhi.first,etaPhi.second);

}

DEFINE_EDM_PLUGIN(edm::MakerPluginFactory,edm::WorkerMaker<L1CaloClusterEGIsolator>,"L1CaloClusterEGIsolator");
DEFINE_FWK_PSET_DESC_FILLER(L1CaloClusterEGIsolator);

