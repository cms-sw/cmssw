// -*- C++ -*-
//
// Package:    AnaMuonCaloCleaner
// Class:      AnaMuonCaloCleaner
// 

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/PatCandidates/interface/Muon.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "TTree.h"
#include "TFile.h"
#include "TH2F.h"


#include <boost/foreach.hpp>

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"


//
// class declaration
//

class AnaMuonCaloCleaner : public edm::EDAnalyzer {
   public:
      explicit AnaMuonCaloCleaner(const edm::ParameterSet&);
      ~AnaMuonCaloCleaner();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      virtual void beginRun(edm::Run const&, edm::EventSetup const&);
      virtual void endRun(edm::Run const&, edm::EventSetup const&);
      virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
      virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
      edm::InputTag colLen_;
      edm::InputTag colDep_;
      edm::InputTag _inputCol;
      int charge_;
      
      std::string getKey(const DetId & det);
      std::vector<std::string> getAllKeys();
      
      
      typedef std::map<int, std::string > TMyMainMap;
      typedef std::map<int, std::map<int, std::string> > TMySubMap;
      typedef std::map<std::string, TH2F * > THistoMap;
      
      
      THistoMap histoMap_;
      TMyMainMap detMap_;
      TMySubMap  subDetMap_;
      
      typedef std::map<std::string, TTree * > TTreeMap;
      typedef std::map<std::string, std::map<std::string, float> > TTreeTempStorageMap;
      TTreeMap treeMap_;
      TTreeTempStorageMap treeStorageMap_;
      
      // ----------member data ---------------------------
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
AnaMuonCaloCleaner::AnaMuonCaloCleaner(const edm::ParameterSet& iConfig):
 colLen_(iConfig.getParameter<edm::InputTag>("colLen")),
 colDep_(iConfig.getParameter<edm::InputTag>("colDep")),
 _inputCol(iConfig.getParameter<edm::InputTag>("selectedMuons")),
 charge_(iConfig.getParameter<int>("charge"))
{
  
  detMap_[DetId::Hcal]="Hcal";
  detMap_[DetId::Ecal]="Ecal";
  
  subDetMap_[DetId::Ecal][EcalBarrel]="EcalBarrel";
  subDetMap_[DetId::Ecal][EcalEndcap]="EcalEndcap";
  subDetMap_[DetId::Ecal][EcalPreshower ]="EcalPreshower";
  subDetMap_[DetId::Ecal][EcalTriggerTower]="EcalTriggerTower";
  subDetMap_[DetId::Ecal][EcalLaserPnDiode]="EcalLaserPnDiode";
  
  subDetMap_[DetId::Hcal][HcalEmpty]="HcalEmpty";
  subDetMap_[DetId::Hcal][HcalBarrel]="HcalBarrel";
  subDetMap_[DetId::Hcal][HcalEndcap]="HcalEndcap";
  subDetMap_[DetId::Hcal][HcalOuter]="HcalOuter";
  subDetMap_[DetId::Hcal][HcalForward]="HcalForward";
  subDetMap_[DetId::Hcal][HcalTriggerTower]="HcalTriggerTower";
  subDetMap_[DetId::Hcal][HcalOther]="HcalOther";

  
  std::vector<std::string> keys = getAllKeys();
  edm::Service<TFileService> fs;
  BOOST_FOREACH(std::string & name, keys){
    histoMap_[name]=fs->make<TH2F>( name.c_str()  , name.c_str(), 1000,  0., 200., 1000, 0, 20. );
    std::string treeName = "tree_" +name ;
    treeMap_[name]=fs->make<TTree>( treeName.c_str()  , treeName.c_str());
    
    treeStorageMap_[name]["pt"] = -999;
    treeStorageMap_[name]["p"] = -999;
    treeStorageMap_[name]["eta"] = -999;
    treeStorageMap_[name]["phi"] = -999;
    treeStorageMap_[name]["len"] = -999;
    treeStorageMap_[name]["val"] = -999;
    treeStorageMap_[name]["charge"] = -999;
    
    BOOST_FOREACH(TTreeTempStorageMap::mapped_type::value_type & entry, treeStorageMap_[name] ){
      treeMap_[name]->Branch(entry.first.c_str(), &entry.second);
    }
  }
  
  //        TTreeMap treeMap_;
  //TTreeTempStorageMap treeStorageMap_;
}


void
AnaMuonCaloCleaner::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  //std::cout << "------------------------------\n";
  edm::Handle< std::vector<reco::Muon> > muonsHandle;
  iEvent.getByLabel(_inputCol, muonsHandle); 
    
  const reco::Muon * myMu = 0;
  BOOST_FOREACH(const reco::Muon & mu, *muonsHandle){
      if (!mu.isGlobalMuon()) continue;
      if (mu.charge()!= charge_) continue;
      myMu = &mu;
      break;
  }
    
  if (myMu==0){
    // std::cout << " XXX AnaMuonCaloCleaner says whoooops" << std::endl;
    return;
  }
  
  typedef std::map<unsigned int,float> TMyCol;
  edm::Handle< TMyCol > hLenghts;
  edm::Handle< TMyCol > hDeposits;
  

  iEvent.getByLabel(colLen_,hLenghts);
  iEvent.getByLabel(colDep_,hDeposits);
  
  
  BOOST_FOREACH(const TMyCol::value_type & entry, *hLenghts){
    DetId det(entry.first);
    float len = entry.second;
    float val = 0;
    std::string name =getKey(det);
    if (hDeposits->find(entry.first) != hDeposits->end())
      val = (hDeposits->find(entry.first))->second;
    //else 
    //  std::cout << "Whoops! empty "<< len << " " << name  << std::endl;

    //std::cout << "XX " << name << " " << det.rawId() << " " << len << " " << val << std::endl;
    histoMap_[name]->Fill(len,val);
    treeStorageMap_[name]["pt"] = myMu->pt();
    treeStorageMap_[name]["p"] = myMu->p();
    treeStorageMap_[name]["eta"] = myMu->eta();
    treeStorageMap_[name]["phi"] = myMu->phi();
    treeStorageMap_[name]["len"] = len;
    treeStorageMap_[name]["val"] = val;
    treeStorageMap_[name]["charge"] = myMu->charge();
    treeMap_[name]->Fill();
  }

  // edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> >    "ecalRecHit"                "EcalRecHitsEE"   "HLT"          EcalRecHitsSorted_ecalRecHit_EcalRecHitsEE_HLT



}
   

std::string AnaMuonCaloCleaner::getKey(const DetId & det){
  return "H_"+detMap_[det.det()]+"_"+subDetMap_[det.det()][det.subdetId()];
}

std::vector<std::string> AnaMuonCaloCleaner::getAllKeys(){
  std::vector<std::string> ret;
  ret.push_back("H__");
  BOOST_FOREACH(TMyMainMap::value_type & entry, detMap_){
    BOOST_FOREACH(TMySubMap::mapped_type::value_type & subEntry, subDetMap_[entry.first]){
      std::string name = "H_"+entry.second+"_"+subEntry.second;
      //std::cout << "XX " << name << std::endl;
      ret.push_back(name);
    }
    
  }
  
  return ret;
}

AnaMuonCaloCleaner::~AnaMuonCaloCleaner()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------


// ------------ method called once each job just before starting event loop  ------------
void 
AnaMuonCaloCleaner::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
AnaMuonCaloCleaner::endJob() 
{

}

// ------------ method called when starting to processes a run  ------------
void 
AnaMuonCaloCleaner::beginRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a run  ------------
void 
AnaMuonCaloCleaner::endRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
void 
AnaMuonCaloCleaner::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void 
AnaMuonCaloCleaner::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
AnaMuonCaloCleaner::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(AnaMuonCaloCleaner);
