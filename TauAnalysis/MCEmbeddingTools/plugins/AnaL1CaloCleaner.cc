// -*- C++ -*-
//
// Package:    AnaL1CaloCleaner
// Class:      AnaL1CaloCleaner
// 

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
//#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/CaloMET.h"

#include "TTree.h"
#include "TFile.h"
#include "Math/VectorUtil.h"

//
// class declaration
//

class AnaL1CaloCleaner: public edm::EDAnalyzer
{
public:
	explicit AnaL1CaloCleaner(const edm::ParameterSet&);
	~AnaL1CaloCleaner();

	static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
private:
	virtual void analyze(const edm::Event&, const edm::EventSetup&);

	edm::InputTag colCaloLengthsMinus_;
	edm::InputTag colCaloLengthsPlus_;
	edm::InputTag colL1ETM_;
	edm::InputTag colCaloMET_;
	edm::InputTag colGenParticles_;
	edm::InputTag colMuons_;

	TTree* tree_;
	Float_t pt_;
	Float_t p_;
	Float_t eta_;
	Float_t phi_;
	Float_t len_;
	Float_t l1_upara_;
	Float_t l1_uperp_;
	Float_t calo_upara_;
	Float_t calo_uperp_;
	Int_t charge_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

std::string getKey(const DetId& det)
{
  std::map<int, std::string > detMap_;
  std::map<int, std::map<int, std::string> > subDetMap_;

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

  return "H_"+detMap_[det.det()]+"_"+subDetMap_[det.det()][det.subdetId()];
}

std::pair<double, double> compMEtProjU(const reco::Candidate::LorentzVector& zP4, double metPx, double metPy, int& errorFlag)
{
  if ( zP4.pt() == 0. ) {
    edm::LogWarning ("compMEtProjU")
      << " Failed to compute projection, because Z0 candidate has zero Pt --> returning dummy solution !!";
    errorFlag = 1;
    return std::pair<double, double>(0., 0.);
  }
  
  double qX = zP4.px();
  double qY = zP4.py();
  double qT = TMath::Sqrt(qX*qX + qY*qY);
  
  double uX = -metPx;//(qX + metPx);
  double uY = -metPy;//(qY + metPy);
  
  double u1 = (uX*qX + uY*qY)/qT;
  double u2 = (uX*qY - uY*qX)/qT;
  
  return std::pair<double, double>(u1, u2);
}

//
// constructors and destructor
//
AnaL1CaloCleaner::AnaL1CaloCleaner(const edm::ParameterSet& iConfig):
	colCaloLengthsMinus_(iConfig.getParameter<edm::InputTag>("caloLengthsMinus")),
	colCaloLengthsPlus_(iConfig.getParameter<edm::InputTag>("caloLengthsPlus")),
	colL1ETM_(iConfig.getParameter<edm::InputTag>("l1ETM")),
	colCaloMET_(iConfig.getParameter<edm::InputTag>("caloMET")),
	colGenParticles_(iConfig.getParameter<edm::InputTag>("genParticles")),
	colMuons_(iConfig.getParameter<edm::InputTag>("muons"))
{
	edm::Service<TFileService> fs;
	tree_ = fs->make<TTree>("L1AnaTree", "L1AnaTree");

	tree_->Branch("pt", &pt_, "pt/F");
	tree_->Branch("p", &p_, "p/F");
	tree_->Branch("eta", &eta_, "eta/F");
	tree_->Branch("phi", &phi_, "phi/F");
	tree_->Branch("len", &len_, "len/F");
	tree_->Branch("l1_upara", &l1_upara_, "l1_upara/F");
	tree_->Branch("l1_uperp", &l1_uperp_, "l1_uperp/F");
	tree_->Branch("calo_upara", &calo_upara_, "calo_upara/F");
	tree_->Branch("calo_uperp", &calo_uperp_, "calo_uperp/F");
	tree_->Branch("charge", &charge_, "charge_/I");
}

void
AnaL1CaloCleaner::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // Find the (only) status-1 genlevel muon
  edm::Handle< std::vector<reco::GenParticle> > genHandle;
  iEvent.getByLabel(colGenParticles_, genHandle);
  const reco::GenParticle* genMuon = NULL;
  for(std::vector<reco::GenParticle>::const_iterator iter = genHandle->begin(); iter != genHandle->end(); ++iter)
  {
    const reco::GenParticle& gen = *iter;
    if(TMath::Abs(gen.pdgId()) != 13) continue;
    if(gen.status() != 1) continue;
    if(genMuon) return; // There should only be one muon, or the whole method does not work
    genMuon = &gen;
  }

  // There should be at least one such muon
  if(!genMuon) return;

  // Next, find the matching reco muon, if any
  edm::Handle< std::vector<reco::Muon> > muonsHandle;
  iEvent.getByLabel(colMuons_, muonsHandle);
  const reco::Muon* recoMuon = NULL;
  for(std::vector<reco::Muon>::const_iterator iter = muonsHandle->begin(); iter != muonsHandle->end(); ++iter)
  {
    const reco::Muon& mu = *iter;
    if(!mu.isGlobalMuon()) continue;
    if(ROOT::Math::VectorUtil::DeltaR(genMuon->p4(), mu.p4()) > 0.1) continue;
    if(recoMuon && ROOT::Math::VectorUtil::DeltaR(genMuon->p4(), mu.p4()) > ROOT::Math::VectorUtil::DeltaR(genMuon->p4(), recoMuon->p4())) continue;
    recoMuon = &mu;
  }

  // We need a reco muon
  if(!recoMuon) return;

  // Next, read the pathlength the muon travelled in the calorimeter
  edm::Handle<std::map<unsigned int,float> > hLengthsMinus, hLengthsPlus;
  iEvent.getByLabel(colCaloLengthsMinus_, hLengthsMinus);
  iEvent.getByLabel(colCaloLengthsPlus_, hLengthsPlus);
  const std::map<unsigned int,float>& caloLengths = recoMuon->charge() < 0 ? *hLengthsMinus : *hLengthsPlus;
  float len = 0.0f;
  for(std::map<unsigned int, float>::const_iterator iter = caloLengths.begin(); iter != caloLengths.end(); ++iter)
  {
    //std::string name = getKey(iter->first);
    //std::cout << name << ": " << iter->second << std::endl;
    len += iter->second;
  }

  // Compute the parallel and perpendicular projection wrt. the genmuon axis
  edm::Handle<l1extra::L1EtMissParticleCollection> hl1Etm;
  iEvent.getByLabel(colL1ETM_, hl1Etm);
  assert(hl1Etm->size() == 1);
  int errorFlag = 0;
  std::pair<double, double> l1_u1u2 = compMEtProjU(genMuon->p4(), (*hl1Etm)[0].px(), (*hl1Etm)[0].py(), errorFlag);
  if(errorFlag) return;

  // Do the same with calo MET for reference
  edm::Handle<std::vector<reco::CaloMET> > hCaloMet;
  iEvent.getByLabel(colCaloMET_, hCaloMet);
  assert(hCaloMet->size() == 1);
  errorFlag = 0;
  std::pair<double, double> calo_u1u2 = compMEtProjU(genMuon->p4(), (*hCaloMet)[0].px(), (*hCaloMet)[0].py(), errorFlag);
  if(errorFlag) return;

  // And finally fill the tree
  pt_ = recoMuon->pt();
  p_ = recoMuon->p();
  eta_ = recoMuon->eta();
  phi_ = recoMuon->phi();
  len_ = len;
  l1_upara_ = l1_u1u2.first;
  l1_uperp_ = l1_u1u2.second;
  calo_upara_ = calo_u1u2.first;
  calo_uperp_ = calo_u1u2.second;
  charge_ = recoMuon->charge();

  tree_->Fill();
}
   
AnaL1CaloCleaner::~AnaL1CaloCleaner()
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
AnaL1CaloCleaner::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(AnaL1CaloCleaner);
