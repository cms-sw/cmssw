#include "TauAnalysis/MCEmbeddingTools/plugins/EmbeddingKineReweightNtupleProducer.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "SimDataFormats/GeneratorProducts/interface/GenFilterInfo.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/Common/interface/View.h"

#include "TauAnalysis/MCEmbeddingTools/interface/embeddingAuxFunctions.h"

#include <TMath.h>
#include <TString.h>

typedef edm::View<reco::Candidate> CandidateView;
typedef edm::View<reco::MET> METView;

int EmbeddingKineReweightNtupleProducer::verbosity_ = 0;

EmbeddingKineReweightNtupleProducer::EmbeddingKineReweightNtupleProducer(const edm::ParameterSet& cfg) 
  : moduleLabel_(cfg.getParameter<std::string>("@module_label")),
    ntuple_(0)
{
  srcGenDiTaus_ = cfg.getParameter<edm::InputTag>("srcGenDiTaus");

  srcGenParticles_ = cfg.getParameter<edm::InputTag>("srcGenParticles"); 
  srcSelectedMuons_ = cfg.getParameter<edm::InputTag>("srcSelectedMuons");

  srcRecLeg1_ = cfg.getParameter<edm::InputTag>("srcRecLeg1");
  srcRecLeg2_ = cfg.getParameter<edm::InputTag>("srcRecLeg2");

  srcRecJets_ = cfg.getParameter<edm::InputTag>("srcRecJets");

  srcGenCaloMEt_ = cfg.getParameter<edm::InputTag>("srcGenCaloMEt");
  srcRecCaloMEt_ = cfg.getParameter<edm::InputTag>("srcRecCaloMEt");
  srcGenPFMEt_ = cfg.getParameter<edm::InputTag>("srcGenPFMEt");
  srcRecPFMEt_ = cfg.getParameter<edm::InputTag>("srcRecPFMEt");

  srcWeights_ = cfg.getParameter<vInputTag>("srcWeights");
  srcGenFilterInfo_ = cfg.getParameter<edm::InputTag>("srcGenFilterInfo");
}

EmbeddingKineReweightNtupleProducer::~EmbeddingKineReweightNtupleProducer()
{
// nothing to be done yet...
}

void EmbeddingKineReweightNtupleProducer::beginJob()
{
//--- create TTree
  edm::Service<TFileService> fs;
  ntuple_ = fs->make<TTree>("embeddingKineReweightNtuple", "embeddingKineReweightNtuple");

//--- add branches storing quantities for computing embeddingKineReweight LUTs
//     1.) for embeddingKineReweightGENtoEmbedded:
  addBranch_EnPxPyPz("genDiTau");
  addBranch_EnPxPyPz("genTau1");
  addBranch_EnPxPyPz("genTau2");  
//     2.) for embeddingKineReweightGENtoREC:
  addBranch_EnPxPyPz("genDiMuon");
  addBranch_EnPxPyPz("genMuonPlus");
  addBranch_EnPxPyPz("genMuonMinus");
  addBranch_EnPxPyPz("recDiMuon");
  addBranch_EnPxPyPz("recMuonPlus");
  addBranch_EnPxPyPz("recMuonMinus");

  addBranch_EnPxPyPz("recLeg1");
  addBranch_EnPxPyPz("recLeg2");

  addBranchI("numJetsRawPtGt20");
  addBranchI("numJetsCorrPtGt20");
  addBranchI("numJetsRawPtGt30");
  addBranchI("numJetsCorrPtGt30");

  addBranch_EnPxPyPz("genCaloMEt");
  addBranch_EnPxPyPz("recCaloMEt");
  addBranch_MEtResProjections("recMinusGenCaloMEt");
  addBranch_EnPxPyPz("genPFMEt");
  addBranch_EnPxPyPz("recPFMEt");
  addBranch_MEtResProjections("recMinusGenPFMEt");

  for ( vInputTag::const_iterator srcWeight = srcWeights_.begin();
	srcWeight != srcWeights_.end(); ++srcWeight ) {
    std::string branchName = Form("%s_%s", srcWeight->label().data(), srcWeight->instance().data());
    addBranchF(branchName.data());
  }
  if ( srcGenFilterInfo_.label() != "" ) {
    addBranchF("genFilterInfo");
    addBranchI("genFilterInfoIsValid");
  }
}

namespace
{
  void findDaughters(const reco::GenParticle* mother, std::vector<const reco::GenParticle*>& daughters, int status)
  {
    unsigned numDaughters = mother->numberOfDaughters();
    for ( unsigned iDaughter = 0; iDaughter < numDaughters; ++iDaughter ) {
      const reco::GenParticle* daughter = mother->daughterRef(iDaughter).get();
      
      if ( status == -1 || daughter->status() == status ) daughters.push_back(daughter);
      
      findDaughters(daughter, daughters, status);
    }
  }
}

void EmbeddingKineReweightNtupleProducer::analyze(const edm::Event& evt, const edm::EventSetup& es) 
{
//--- reset all branches
  for ( branchMap::iterator branch = branches_.begin();
	branch != branches_.end(); ++branch ) {
    branch->second.valueF_ = 0.;
    branch->second.valueI_ = 0;
  }

//--- set branches containing generator level Z->tautau information
  const reco::Candidate* genDiTau_ref = 0;
  if ( srcGenDiTaus_.label() != "" ) {
    edm::Handle<CandidateView> genDiTaus;
    evt.getByLabel(srcGenDiTaus_, genDiTaus);

    for ( CandidateView::const_iterator genDiTau = genDiTaus->begin();
	  genDiTau != genDiTaus->end(); ++genDiTau ) {
      const reco::CompositeCandidate* genDiTau_composite = dynamic_cast<const reco::CompositeCandidate*>(&(*genDiTau));
      if ( !(genDiTau_composite && genDiTau_composite->numberOfDaughters() == 2) ) {
	std::cerr << "genDiTau has " << genDiTau_composite->numberOfDaughters() << " daughters --> skipping !!" << std::endl;
	continue;	
      }
      
      const reco::Candidate* genTau1 = genDiTau_composite->daughter(0);
      const reco::Candidate* genTau2 = genDiTau_composite->daughter(1);
      if ( !(genTau1 && genTau2) ) {
	std::cerr << "Failed to find daughters of genDiTau --> skipping !!" << std::endl;
	continue;
      }
      const reco::Candidate* genTauPlus  = 0;
      const reco::Candidate* genTauMinus = 0;
      if ( genTau1->charge() > +0.5 && genTau2->charge() < -0.5 ) {
	genTauPlus  = genTau1;
	genTauMinus = genTau2;
      } else if ( genTau1->charge() < -0.5 && genTau2->charge() > +0.5 ) {
	genTauPlus  = genTau2;
	genTauMinus = genTau1;
      } else {
	std::cerr << "Failed to find daughters of genDiTau --> skipping !!" << std::endl;
	continue;
      }
      
      setValue_EnPxPyPz("genDiTau", genDiTau->p4());
      setValue_EnPxPyPz("genTauPlus", genTauPlus->p4());
      setValue_EnPxPyPz("genTauMinus", genTauMinus->p4());
      genDiTau_ref = &(*genDiTau);
    }
  }

  if ( !genDiTau_ref ) {
    std::cerr << "Failed to find genDiTau --> skipping !!" << std::endl;
    return;
  }

//--- set branches containing generator level Z->mumu information
  if ( srcGenParticles_.label() != "" ) {
    edm::Handle<reco::GenParticleCollection> genParticles;
    evt.getByLabel(srcGenParticles_, genParticles);
    
    for ( reco::GenParticleCollection::const_iterator genParticle = genParticles->begin();
	  genParticle != genParticles->end(); ++genParticle ) {
      if ( TMath::Abs(genParticle->pdgId()) == 23 ) {
	std::vector<const reco::GenParticle*> daughters;
	findDaughters(&(*genParticle), daughters, -1);
	const reco::GenParticle* genMuPlus   = 0;
	const reco::GenParticle* genMuMinus  = 0;
	for ( std::vector<const reco::GenParticle*>::const_iterator daughter = daughters.begin();
	      daughter != daughters.end(); ++daughter ) {
	  if ( (*daughter)->pdgId() == -13 ) genMuPlus   = (*daughter);
	  if ( (*daughter)->pdgId() == +13 ) genMuMinus  = (*daughter);
	}
	if ( genMuPlus && genMuMinus ) {
	  setValue_EnPxPyPz("genDiMuon", genParticle->p4());
	  setValue_EnPxPyPz("genMuonPlus", genMuPlus->p4());
	  setValue_EnPxPyPz("genMuonMinus", genMuMinus->p4());
	  
//--- set branches containing information on reconstructed Z->mumu decays
	  if ( srcSelectedMuons_.label() != "" ) {
	    std::vector<reco::CandidateBaseRef> selMuons = getSelMuons(evt, srcSelectedMuons_);
	    const reco::CandidateBaseRef recMuPlus  = getTheMuPlus(selMuons);
	    const reco::CandidateBaseRef recMuMinus = getTheMuMinus(selMuons);
	    if ( recMuPlus.isNonnull() && recMuMinus.isNonnull() ) {
	      setValue_EnPxPyPz("recDiMuon", recMuPlus->p4() + recMuMinus->p4());
	      setValue_EnPxPyPz("recMuonPlus", recMuPlus->p4());
	      setValue_EnPxPyPz("recMuonMinus", recMuMinus->p4());
	    }
	  }
	}
      }
    }
  }

  edm::Handle<CandidateView> recLeg1;
  evt.getByLabel(srcRecLeg1_, recLeg1);
  if ( recLeg1->size() >= 1 ) {
    setValue_EnPxPyPz("recLeg1", recLeg1->at(0).p4());
  }
  edm::Handle<CandidateView> recLeg2;
  evt.getByLabel(srcRecLeg2_, recLeg2);
  if ( srcRecLeg2_ == srcRecLeg1_ && recLeg2->size() >= 2 ) {
    setValue_EnPxPyPz("recLeg2", recLeg1->at(1).p4());
  } else if ( recLeg2->size() >= 1 ) {
    setValue_EnPxPyPz("recLeg2", recLeg2->at(0).p4());
  }

  edm::Handle<pat::JetCollection> recJets;
  evt.getByLabel(srcRecJets_, recJets);
  int numJetsRawPtGt20  = 0;
  int numJetsCorrPtGt20 = 0;
  int numJetsRawPtGt30  = 0;
  int numJetsCorrPtGt30 = 0;
  for ( pat::JetCollection::const_iterator recJet = recJets->begin();
	recJet != recJets->end(); ++recJet ) {
    
    reco::Candidate::LorentzVector rawJetP4 = recJet->correctedP4("Uncorrected");
    double rawJetPt = rawJetP4.pt();
    double rawJetAbsEta = TMath::Abs(rawJetP4.eta());
    if ( rawJetAbsEta < 4.5 ) {
      if ( rawJetPt > 20. ) ++numJetsRawPtGt20;
      if ( rawJetPt > 30. ) ++numJetsRawPtGt30;
    }
    
    reco::Candidate::LorentzVector corrJetP4 = recJet->p4();
    double corrJetPt = corrJetP4.pt();
    double corrJetAbsEta = TMath::Abs(corrJetP4.eta());
    if ( corrJetAbsEta < 4.5 ) {
      if ( corrJetPt > 20. ) ++numJetsCorrPtGt20;
      if ( corrJetPt > 30. ) ++numJetsCorrPtGt30;
    }
  }
  setValueI("numJetsRawPtGt20", numJetsRawPtGt20);
  setValueI("numJetsCorrPtGt20", numJetsCorrPtGt20);
  setValueI("numJetsRawPtGt30", numJetsRawPtGt30);
  setValueI("numJetsCorrPtGt30", numJetsCorrPtGt30);

  if ( genDiTau_ref ) {
    edm::Handle<METView> genCaloMET;
    evt.getByLabel(srcGenCaloMEt_, genCaloMET);
    const reco::Candidate::LorentzVector& genCaloMEtP4 = genCaloMET->front().p4();
    setValue_EnPxPyPz("genCaloMEt", genCaloMEtP4);
    edm::Handle<METView> recCaloMET;
    evt.getByLabel(srcRecCaloMEt_, recCaloMET);
    const reco::Candidate::LorentzVector& recCaloMEtP4 = recCaloMET->front().p4();
    //std::cout << "<EmbeddingKineReweightNtupleProducer>:" << std::endl;
    //std::cout << " recCaloMEt(" << srcRecCaloMEt_.label() << "): Pt = " << recCaloMEtP4.pt() << ", phi = " << recCaloMEtP4.phi() 
    //	        << " (Px = " << recCaloMEtP4.px() << ", Py = " << recCaloMEtP4.py() << ")" << std::endl;
    setValue_EnPxPyPz("recCaloMEt", recCaloMEtP4);
    setValue_MEtResProjections("recMinusGenCaloMEt", genCaloMEtP4, recCaloMEtP4, genDiTau_ref->p4());

    edm::Handle<METView> genPFMET;
    evt.getByLabel(srcGenPFMEt_, genPFMET);
    const reco::Candidate::LorentzVector& genPFMEtP4 = genPFMET->front().p4();
    setValue_EnPxPyPz("genPFMEt", genPFMEtP4);
    edm::Handle<METView> recPFMET;
    evt.getByLabel(srcRecPFMEt_, recPFMET);
    const reco::Candidate::LorentzVector& recPFMEtP4 = recPFMET->front().p4();
    setValue_EnPxPyPz("recPFMEt", recPFMEtP4);
    setValue_MEtResProjections("recMinusGenPFMEt", genPFMEtP4, recPFMEtP4, genDiTau_ref->p4());
  }

//--- set branches containing event weight information 
  for ( vInputTag::const_iterator srcWeight = srcWeights_.begin();
	srcWeight != srcWeights_.end(); ++srcWeight ) {
    edm::Handle<double> weight;
    evt.getByLabel(*srcWeight, weight);
    std::string branchName = Form("%s_%s", srcWeight->label().data(), srcWeight->instance().data());
    setValueF(branchName.data(), *weight);
  }
  if ( srcGenFilterInfo_.label() != "" ) {
    edm::Handle<GenFilterInfo> genFilterInfo;
    evt.getByLabel(srcGenFilterInfo_, genFilterInfo);
    double genFilterInfo_weight = 1.;
    bool genFilterInfo_isValid = false;
    if ( genFilterInfo->numEventsTried() > 0 ) {
      genFilterInfo_weight = genFilterInfo->filterEfficiency();
      genFilterInfo_isValid = true;
    }
    setValueF("genFilterInfo", genFilterInfo_weight);
    setValueI("genFilterInfoIsValid", genFilterInfo_isValid);
  }

//--- finally, fill all computed quantities into TTree
  assert(ntuple_);
  ntuple_->Fill();
}

void EmbeddingKineReweightNtupleProducer::addBranchF(const std::string& name) 
{
  //std::cout << "<EmbeddingKineReweightNtupleProducer::addBranchF>:" << std::endl;
  //std::cout << " moduleLabel = " << moduleLabel_ << std::endl;
  //std::cout << " branchName = " << name << std::endl;
  assert(branches_.count(name) == 0);
  std::string name_and_format = name + "/F";
  ntuple_->Branch(name.c_str(), &branches_[name].valueF_, name_and_format.c_str());
}

void EmbeddingKineReweightNtupleProducer::addBranchI(const std::string& name) 
{
  //std::cout << "<EmbeddingKineReweightNtupleProducer::addBranchI>:" << std::endl;
  //std::cout << " moduleLabel = " << moduleLabel_ << std::endl;
  //std::cout << " branchName = " << name << std::endl;
  assert(branches_.count(name) == 0);
  std::string name_and_format = name + "/I";
  ntuple_->Branch(name.c_str(), &branches_[name].valueI_, name_and_format.c_str());
}

void EmbeddingKineReweightNtupleProducer::printBranches(std::ostream& stream)
{
  stream << "<EmbeddingKineReweightNtupleProducer::printBranches>:" << std::endl;
  stream << " registered branches for module = " << moduleLabel_ << std::endl;
  for ( branchMap::const_iterator branch = branches_.begin();
        branch != branches_.end(); ++branch ) {
    stream << " " << branch->first << std::endl;
  }
  stream << std::endl;
}

void EmbeddingKineReweightNtupleProducer::setValueF(const std::string& name, double value) 
{
  if ( verbosity_ ) std::cout << "branch = " << name << ": value = " << value << std::endl;
  branchMap::iterator branch = branches_.find(name);
  if ( branch != branches_.end() ) {
    branch->second.valueF_ = value;
  } else {
    throw cms::Exception("InvalidParameter") 
      << "No branch with name = " << name << " defined !!\n";
  }
}

void EmbeddingKineReweightNtupleProducer::setValueI(const std::string& name, int value) 
{
  if ( verbosity_ ) std::cout << "branch = " << name << ": value = " << value << std::endl;
  branchMap::iterator branch = branches_.find(name);
  if ( branch != branches_.end() ) {
    branch->second.valueI_ = value;
  } else {
    throw cms::Exception("InvalidParameter") 
      << "No branch with name = " << name << " defined !!\n";
  }
}

//
//-------------------------------------------------------------------------------
//

void EmbeddingKineReweightNtupleProducer::addBranch_EnPxPyPz(const std::string& name) 
{
  addBranchF(std::string(name).append("En"));
  addBranchF(std::string(name).append("Px"));
  addBranchF(std::string(name).append("Py"));
  addBranchF(std::string(name).append("Pz"));
  addBranchF(std::string(name).append("Pt"));
  addBranchF(std::string(name).append("Eta"));
  addBranchF(std::string(name).append("Phi"));
  addBranchF(std::string(name).append("M"));
  addBranchI(std::string(name).append("IsValid"));
}

void EmbeddingKineReweightNtupleProducer::addBranch_MEtResProjections(const std::string& name)
{
  addBranchF(std::string(name).append("ParlZ"));
  addBranchF(std::string(name).append("PerpZ"));
}

//
//-------------------------------------------------------------------------------
//

void EmbeddingKineReweightNtupleProducer::setValue_EnPxPyPz(const std::string& name, const reco::Candidate::LorentzVector& p4)
{
  setValueF(std::string(name).append("En"), p4.E());
  setValueF(std::string(name).append("Px"), p4.px());
  setValueF(std::string(name).append("Py"), p4.py());
  setValueF(std::string(name).append("Pz"), p4.pz());
  setValueF(std::string(name).append("Pt"), p4.pt());
  setValueF(std::string(name).append("Eta"), p4.eta());
  setValueF(std::string(name).append("Phi"), p4.phi());
  setValueF(std::string(name).append("M"), p4.M());
  setValueI(std::string(name).append("IsValid"), true);
}

void EmbeddingKineReweightNtupleProducer::setValue_MEtResProjections(const std::string& name, 
								     const reco::Candidate::LorentzVector& genMEtP4, const reco::Candidate::LorentzVector& recMEtP4, 
								     const reco::Candidate::LorentzVector& zP4)
{
  if ( zP4.pt() > 0. ) {
    double qX = zP4.px();
    double qY = zP4.py();
    double qT = TMath::Sqrt(qX*qX + qY*qY);
    double dX = recMEtP4.px() - genMEtP4.px();
    double dY = recMEtP4.py() - genMEtP4.py();
    double dParl = (dX*qX + dY*qY)/qT;
    double dPerp = (dX*qY - dY*qX)/qT;
    setValueF(std::string(name).append("ParlZ"), dParl);
    setValueF(std::string(name).append("PerpZ"), dPerp);
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(EmbeddingKineReweightNtupleProducer);
