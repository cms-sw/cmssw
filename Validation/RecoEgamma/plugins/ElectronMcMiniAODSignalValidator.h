#ifndef Validation_RecoEgamma_ElectronMcSignalValidatorMiniAOD_h 
#define Validation_RecoEgamma_ElectronMcSignalValidatorMiniAOD_h

#include "DQMOffline/EGamma/interface/ElectronDqmAnalyzerBase.h"

// system include files
//#include <memory>

// user include files
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/PatCandidates/interface/Electron.h"

// user include files

//
// class declaration
//
class ElectronMcSignalValidatorMiniAOD : public ElectronDqmAnalyzerBase {
   public:
      explicit ElectronMcSignalValidatorMiniAOD(const edm::ParameterSet&);
      virtual ~ElectronMcSignalValidatorMiniAOD();
      bool isAncestor(const reco::Candidate * ancestor, const reco::Candidate * particle);

   private:
      virtual void bookHistograms( DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

      // ----------member data ---------------------------
      edm::EDGetTokenT<edm::View<reco::GenParticle> > mcTruthCollection_; // prunedGenParticles
      edm::EDGetTokenT<pat::ElectronCollection> electronToken_; // slimmedElectrons

      edm::EDGetTokenT<edm::ValueMap<float> > ValueMaps_ChargedHadrons_;
      edm::EDGetTokenT<edm::ValueMap<float> > ValueMaps_NeutralHadrons_;
      edm::EDGetTokenT<edm::ValueMap<float> > ValueMaps_Photons_;
      float pt_;
 
      double maxPt_;
      double maxAbsEta_;
      double deltaR_, deltaR2_;
      std::vector<int> matchingIDs_;
      std::vector<int> matchingMotherIDs_;
      std::string outputInternalPath_ ;

      float sumChargedHadronPt_recomp;
      float sumNeutralHadronPt_recomp;
      float sumPhotonPt_recomp;
      float relisoChargedHadronPt_recomp;
      float relisoNeutralHadronPt_recomp;
      float relisoPhotonPt_recomp;

      // histos limits and binning

      int xyz_nbin ;
      int pt_nbin ; int pt2D_nbin ; int pteff_nbin ; double pt_max ;
      int fhits_nbin ; double fhits_max ;
      int eta_nbin ; int eta2D_nbin ; double eta_min ; double eta_max ;
      int deta_nbin ; double deta_min ; double deta_max ;
      int detamatch_nbin ; int detamatch2D_nbin ; double detamatch_min ; double detamatch_max ;
      int phi_nbin ; int phi2D_nbin ; double phi_min ; double phi_max ;
      int dphi_nbin ; double dphi_min ; double dphi_max ;
      int dphimatch_nbin ; int    dphimatch2D_nbin ; double dphimatch_min ; double dphimatch_max ;
      int mee_nbin ; double mee_min ; double mee_max ;
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

    MonitorElement *h2_ele_foundHitsVsEta;
    MonitorElement *h2_ele_foundHitsVsEta_mAOD;

    MonitorElement *h2_ele_PoPtrueVsEta;
    MonitorElement *h2_ele_sigmaIetaIetaVsPt;

    MonitorElement *h1_ele_HoE_mAOD;
    MonitorElement *h1_ele_HoE_mAOD_barrel;
    MonitorElement *h1_ele_HoE_mAOD_endcaps;
    MonitorElement *h1_ele_mee_all;
    MonitorElement *h1_ele_mee_os;

    MonitorElement *h1_ele_dEtaSc_propVtx_mAOD;
    MonitorElement *h1_ele_dEtaSc_propVtx_mAOD_barrel;
    MonitorElement *h1_ele_dEtaSc_propVtx_mAOD_endcaps;
    MonitorElement *h1_ele_dPhiCl_propOut_mAOD;
    MonitorElement *h1_ele_dPhiCl_propOut_mAOD_barrel;
    MonitorElement *h1_ele_dPhiCl_propOut_mAOD_endcaps;

    MonitorElement *h1_ele_fbrem_mAOD;
    MonitorElement *h1_ele_fbrem_mAOD_barrel;
    MonitorElement *h1_ele_fbrem_mAOD_endcaps;

	// -- pflow over pT
	MonitorElement *h1_ele_chargedHadronRelativeIso_mAOD;
	MonitorElement *h1_ele_chargedHadronRelativeIso_mAOD_barrel;
    MonitorElement *h1_ele_chargedHadronRelativeIso_mAOD_endcaps;
	MonitorElement *h1_ele_neutralHadronRelativeIso_mAOD;
	MonitorElement *h1_ele_neutralHadronRelativeIso_mAOD_barrel;
    MonitorElement *h1_ele_neutralHadronRelativeIso_mAOD_endcaps;
	MonitorElement *h1_ele_photonRelativeIso_mAOD;
	MonitorElement *h1_ele_photonRelativeIso_mAOD_barrel;
    MonitorElement *h1_ele_photonRelativeIso_mAOD_endcaps;

	MonitorElement *h1_ele_chargedHadronRelativeIso_mAOD_recomp;
	MonitorElement *h1_ele_neutralHadronRelativeIso_mAOD_recomp;
    MonitorElement *h1_ele_photonRelativeIso_mAOD_recomp;    
};

#endif
