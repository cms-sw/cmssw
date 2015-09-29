#ifndef Validation_RecoEgamma_ElectronMcMiniAODSignalValidator_h 
#define Validation_RecoEgamma_ElectronMcMiniAODSignalValidator_h

#include "DQMOffline/EGamma/interface/ElectronDqmAnalyzerBase.h"

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"

#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

//
// class declaration
//
class ElectronMcMiniAODSignalValidator : public ElectronDqmAnalyzerBase {
   public:
      explicit ElectronMcMiniAODSignalValidator(const edm::ParameterSet&);
      virtual ~ElectronMcMiniAODSignalValidator();
      bool isAncestor(const reco::Candidate * ancestor, const reco::Candidate * particle);
      
   private:
      virtual void bookHistograms( DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

      // ----------member data ---------------------------
      edm::EDGetTokenT<reco::VertexCollection> vtxToken_;
      edm::EDGetTokenT<edm::View<reco::GenParticle> > mcTruthCollection_;
      
      edm::EDGetTokenT<pat::ElectronCollection> electronToken_;
 
      double maxPt_;
      double maxAbsEta_;
      double deltaR_;
      std::vector<int> matchingIDs_;
      std::vector<int> matchingMotherIDs_;
      std::string outputInternalPath_ ;

      // histos limits and binning

      int xyz_nbin ;
      int pt_nbin ; int pt2D_nbin ; int pteff_nbin ; double pt_max ;
      int eta_nbin ; int eta2D_nbin ; double eta_min ; double eta_max ;
      int hoe_nbin ; double hoe_min ; double hoe_max ;
      int poptrue_nbin ; double poptrue_min ; double poptrue_max ;
      bool set_EfficiencyFlag ; bool set_StatOverflowFlag ;
      
    // histos

    MonitorElement *h1_recEleNum;

    MonitorElement *h1_ele_vertexPt;
    MonitorElement *h1_ele_vertexEta;
    MonitorElement *h1_ele_vertexPt_nocut;

    MonitorElement *h1_scl_SigIEtaIEta_mAOD;
    MonitorElement *h1_scl_SigIEtaIEta_mAOD_barrel;
    MonitorElement *h1_scl_SigIEtaIEta_mAOD_endcaps;

    MonitorElement *h2_ele_PoPtrueVsEta;
    MonitorElement *h2_ele_sigmaIetaIetaVsPt;

    MonitorElement *h1_ele_HoE_mAOD;
    MonitorElement *h1_ele_HoE_mAOD_barrel;
    MonitorElement *h1_ele_HoE_mAOD_endcaps;

    MonitorElement *h1_ele_fbrem_mAOD;
    MonitorElement *h1_ele_fbrem_barrel_mAOD;
    MonitorElement *h1_ele_fbrem_endcaps_mAOD;

	// -- pflow over pT
	MonitorElement *h1_ele_chargedHadronRelativeIso_mAOD;
	MonitorElement *h1_ele_chargedHadronRelativeIso_barrel_mAOD;
    MonitorElement *h1_ele_chargedHadronRelativeIso_endcaps_mAOD;
	MonitorElement *h1_ele_neutralHadronRelativeIso_mAOD;
	MonitorElement *h1_ele_neutralHadronRelativeIso_barrel_mAOD;
    MonitorElement *h1_ele_neutralHadronRelativeIso_endcaps_mAOD;
	MonitorElement *h1_ele_photonRelativeIso_mAOD;
	MonitorElement *h1_ele_photonRelativeIso_barrel_mAOD;
    MonitorElement *h1_ele_photonRelativeIso_endcaps_mAOD;
    
};

#endif
