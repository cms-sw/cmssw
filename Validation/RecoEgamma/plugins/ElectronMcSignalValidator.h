
#ifndef Validation_RecoEgamma_ElectronMcSignalValidator_h
#define Validation_RecoEgamma_ElectronMcSignalValidator_h

#include "DQMOffline/EGamma/interface/ElectronDqmAnalyzerBase.h"
//#include "Validation/RecoEgamma/interface/ElectronValidator.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
class MagneticField;

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

namespace reco {class BeamSpot;}

class ElectronMcSignalValidator : public ElectronDqmAnalyzerBase
 {
  public:

    explicit ElectronMcSignalValidator( const edm::ParameterSet & conf ) ;
    virtual ~ElectronMcSignalValidator() ;
    virtual void book() ;
    virtual void analyze( const edm::Event& e, const edm::EventSetup & c ) ;

  private:

    edm::EDGetTokenT<reco::GenParticleCollection> mcTruthCollection_;
    edm::EDGetTokenT<reco::GsfElectronCollection> electronCollection_;
    edm::EDGetTokenT<reco::GsfElectronCoreCollection> electronCoreCollection_;
    edm::EDGetTokenT<reco::GsfTrackCollection> electronTrackCollection_;
    edm::EDGetTokenT<reco::ElectronSeedCollection> electronSeedCollection_;
    edm::EDGetTokenT<reco::BeamSpot> beamSpotTag_ ;
    bool readAOD_;
    //std::string outputFile_ ;

    edm::EDGetTokenT<edm::ValueMap<double> > isoFromDepsTk03Tag_;
    edm::EDGetTokenT<edm::ValueMap<double> > isoFromDepsTk04Tag_;
    edm::EDGetTokenT<edm::ValueMap<double> > isoFromDepsEcalFull03Tag_;
    edm::EDGetTokenT<edm::ValueMap<double> > isoFromDepsEcalFull04Tag_;
    edm::EDGetTokenT<edm::ValueMap<double> > isoFromDepsEcalReduced03Tag_;
    edm::EDGetTokenT<edm::ValueMap<double> > isoFromDepsEcalReduced04Tag_;
    edm::EDGetTokenT<edm::ValueMap<double> > isoFromDepsHcal03Tag_;
    edm::EDGetTokenT<edm::ValueMap<double> > isoFromDepsHcal04Tag_;

    edm::ESHandle<TrackerGeometry> pDD ;
    edm::ESHandle<MagneticField> theMagField ;

    float mcEnergy[10], mcEta[10], mcPhi[10], mcPt[10], mcQ[10] ;
    float superclusterEnergy[10], superclusterEta[10], superclusterPhi[10], superclusterEt[10] ;
    float seedMomentum[10], seedEta[10], seedPhi[10], seedPt[10], seedQ[10] ;

    double maxPt_;
    double maxAbsEta_;
    double deltaR_;
    std::vector<int> matchingIDs_;
    std::vector<int> matchingMotherIDs_;

    // histos limits and binning

    int xyz_nbin ;
    int p_nbin ; int p2D_nbin ; double p_max ;
    int pt_nbin ; int pt2D_nbin ; int pteff_nbin ; double pt_max ;
    int fhits_nbin ; double fhits_max ;
    int lhits_nbin ; double lhits_max ;
    int eta_nbin ; int eta2D_nbin ; double eta_min ; double eta_max ;
    int deta_nbin ; double deta_min ; double deta_max ;
    int detamatch_nbin ; int detamatch2D_nbin ; double detamatch_min ; double detamatch_max ;
    int phi_nbin ; int phi2D_nbin ; double phi_min ; double phi_max ;
    int dphi_nbin ; double dphi_min ; double dphi_max ;
    int dphimatch_nbin ; int    dphimatch2D_nbin ; double dphimatch_min ; double dphimatch_max ;
    int eop_nbin ; int eop2D_nbin ; double eop_max ; double eopmaxsht ;
    int mee_nbin ; double mee_min ; double mee_max ;
    int hoe_nbin ; double hoe_min ; double hoe_max ;
    int poptrue_nbin ; double poptrue_min ; double poptrue_max ;
    int error_nbin ; double enerror_max ;

    // histos

    MonitorElement *h1_mcNum;
    MonitorElement *h1_eleNum;
    MonitorElement *h1_gamNum;

    MonitorElement *h1_recEleNum;
    MonitorElement *h1_recCoreNum;
    MonitorElement *h1_recTrackNum;
    MonitorElement *h1_recSeedNum;

    MonitorElement *h1_mc_Eta;
    MonitorElement *h1_mc_AbsEta;
    MonitorElement *h1_mc_P;
    MonitorElement *h1_mc_Pt;
    MonitorElement *h1_mc_Phi;
    MonitorElement *h1_mc_Z;
    MonitorElement *h2_mc_PtEta;

    MonitorElement *h1_mc_Eta_matched;
    MonitorElement *h1_mc_AbsEta_matched;
    MonitorElement *h1_mc_Pt_matched;
    MonitorElement *h1_mc_Phi_matched;
    MonitorElement *h1_mc_Z_matched;
    MonitorElement *h2_mc_PtEta_matched;

    MonitorElement *h1_mc_Eta_matched_qmisid;
    MonitorElement *h1_mc_AbsEta_matched_qmisid;
    MonitorElement *h1_mc_Pt_matched_qmisid;
    MonitorElement *h1_mc_Phi_matched_qmisid;
    MonitorElement *h1_mc_Z_matched_qmisid;
    MonitorElement *h1_ele_chargeInfo;

    MonitorElement *h1_ele_EoverP_all;
    MonitorElement *h1_ele_EoverP_all_barrel;
    MonitorElement *h1_ele_EoverP_all_endcaps;
    MonitorElement *h1_ele_EseedOP_all;
    MonitorElement *h1_ele_EseedOP_all_barrel;
    MonitorElement *h1_ele_EseedOP_all_endcaps;
    MonitorElement *h1_ele_EoPout_all;
    MonitorElement *h1_ele_EoPout_all_barrel;
    MonitorElement *h1_ele_EoPout_all_endcaps;
    MonitorElement *h1_ele_EeleOPout_all;
    MonitorElement *h1_ele_EeleOPout_all_barrel;
    MonitorElement *h1_ele_EeleOPout_all_endcaps;
    MonitorElement *h1_ele_dEtaSc_propVtx_all;
    MonitorElement *h1_ele_dEtaSc_propVtx_all_barrel;
    MonitorElement *h1_ele_dEtaSc_propVtx_all_endcaps;
    MonitorElement *h1_ele_dPhiSc_propVtx_all;
    MonitorElement *h1_ele_dPhiSc_propVtx_all_barrel;
    MonitorElement *h1_ele_dPhiSc_propVtx_all_endcaps;
    MonitorElement *h1_ele_dEtaCl_propOut_all;
    MonitorElement *h1_ele_dEtaCl_propOut_all_barrel;
    MonitorElement *h1_ele_dEtaCl_propOut_all_endcaps;
    MonitorElement *h1_ele_dPhiCl_propOut_all;
    MonitorElement *h1_ele_dPhiCl_propOut_all_barrel;
    MonitorElement *h1_ele_dPhiCl_propOut_all_endcaps;
    MonitorElement *h1_ele_TIP_all;
    MonitorElement *h1_ele_TIP_all_barrel;
    MonitorElement *h1_ele_TIP_all_endcaps;
    MonitorElement *h1_ele_HoE_all;
    MonitorElement *h1_ele_HoE_all_barrel;
    MonitorElement *h1_ele_HoE_all_endcaps;
    MonitorElement *h1_ele_HoE_bc_all;
    MonitorElement *h1_ele_vertexEta_all;
    MonitorElement *h1_ele_vertexPt_all;
    MonitorElement *h1_ele_Et_all;
    MonitorElement *h1_ele_mee_all;
    MonitorElement *h1_ele_mee_os;
    MonitorElement *h1_ele_mee_os_ebeb;
    MonitorElement *h1_ele_mee_os_ebee;
    MonitorElement *h1_ele_mee_os_eeee;
    MonitorElement *h1_ele_mee_os_gg;
    MonitorElement *h1_ele_mee_os_gb;
    MonitorElement *h1_ele_mee_os_bb;

    MonitorElement *h2_ele_E2mnE1vsMee_all;
    MonitorElement *h2_ele_E2mnE1vsMee_egeg_all;

    MonitorElement *h1_ele_charge;
    MonitorElement *h2_ele_chargeVsEta;
    MonitorElement *h2_ele_chargeVsPhi;
    MonitorElement *h2_ele_chargeVsPt;
    MonitorElement *h1_ele_vertexP;
    MonitorElement *h1_ele_vertexPt;
    MonitorElement *h1_ele_Et;
    MonitorElement *h2_ele_vertexPtVsEta;
    MonitorElement *h2_ele_vertexPtVsPhi;
    MonitorElement *h1_ele_vertexPt_5100;
    MonitorElement *h1_ele_vertexEta;
    MonitorElement *h2_ele_vertexEtaVsPhi;
    MonitorElement *h1_ele_vertexAbsEta;
    MonitorElement *h1_ele_vertexPhi;
    MonitorElement *h1_ele_vertexX;
    MonitorElement *h1_ele_vertexY;
    MonitorElement *h1_ele_vertexZ;
    MonitorElement *h1_ele_vertexTIP;
    MonitorElement *h2_ele_vertexTIPVsEta;
    MonitorElement *h2_ele_vertexTIPVsPhi;
    MonitorElement *h2_ele_vertexTIPVsPt;
    MonitorElement *h1_ele_ecalEnergyError;
    MonitorElement *h1_ele_ecalEnergyError_barrel;
    MonitorElement *h1_ele_ecalEnergyError_endcaps;
    MonitorElement *h1_ele_combinedP4Error;
    MonitorElement *h1_ele_combinedP4Error_barrel;
    MonitorElement *h1_ele_combinedP4Error_endcaps;

    MonitorElement *h1_scl_En;
    MonitorElement *h1_scl_EoEtrue_barrel;
    MonitorElement *h1_scl_EoEtrue_endcaps;
    MonitorElement *h1_scl_EoEtrue_barrel_eg;
    MonitorElement *h1_scl_EoEtrue_endcaps_eg;
    MonitorElement *h1_scl_EoEtrue_barrel_etagap;
    MonitorElement *h1_scl_EoEtrue_barrel_phigap;
    MonitorElement *h1_scl_EoEtrue_ebeegap;
    MonitorElement *h1_scl_EoEtrue_endcaps_deegap;
    MonitorElement *h1_scl_EoEtrue_endcaps_ringgap;
    MonitorElement *h1_scl_EoEtrue_barrel_new;
    MonitorElement *h1_scl_EoEtrue_endcaps_new;
    MonitorElement *h1_scl_EoEtrue_barrel_new_eg;
    MonitorElement *h1_scl_EoEtrue_endcaps_new_eg;
    MonitorElement *h1_scl_EoEtrue_barrel_new_etagap;
    MonitorElement *h1_scl_EoEtrue_barrel_new_phigap;
    MonitorElement *h1_scl_EoEtrue_ebeegap_new;
    MonitorElement *h1_scl_EoEtrue_endcaps_new_deegap;
    MonitorElement *h1_scl_EoEtrue_endcaps_new_ringgap;
    MonitorElement *h1_scl_Et;
    MonitorElement *h2_scl_EtVsEta;
    MonitorElement *h2_scl_EtVsPhi;
    MonitorElement *h2_scl_EtaVsPhi;
    MonitorElement *h1_scl_Eta;
    MonitorElement *h1_scl_Phi;

    MonitorElement *h2_scl_EoEtruePfVsEg ;

    MonitorElement *h1_scl_SigEtaEta;
    MonitorElement *h1_scl_SigEtaEta_barrel;
    MonitorElement *h1_scl_SigEtaEta_endcaps;
    MonitorElement *h1_scl_SigIEtaIEta;
    MonitorElement *h1_scl_SigIEtaIEta_barrel;
    MonitorElement *h1_scl_SigIEtaIEta_endcaps;
    MonitorElement *h1_scl_E1x5;
    MonitorElement *h1_scl_E1x5_barrel;
    MonitorElement *h1_scl_E1x5_endcaps;
    MonitorElement *h1_scl_E2x5max;
    MonitorElement *h1_scl_E2x5max_barrel;
    MonitorElement *h1_scl_E2x5max_endcaps;
    MonitorElement *h1_scl_E5x5;
    MonitorElement *h1_scl_E5x5_barrel;
    MonitorElement *h1_scl_E5x5_endcaps;
    MonitorElement *h1_scl_SigEtaEta_eg;
    MonitorElement *h1_scl_SigEtaEta_eg_barrel;
    MonitorElement *h1_scl_SigEtaEta_eg_endcaps;
    MonitorElement *h1_scl_SigIEtaIEta_eg;
    MonitorElement *h1_scl_SigIEtaIEta_eg_barrel;
    MonitorElement *h1_scl_SigIEtaIEta_eg_endcaps;
    MonitorElement *h1_scl_E1x5_eg;
    MonitorElement *h1_scl_E1x5_eg_barrel;
    MonitorElement *h1_scl_E1x5_eg_endcaps;
    MonitorElement *h1_scl_E2x5max_eg;
    MonitorElement *h1_scl_E2x5max_eg_barrel;
    MonitorElement *h1_scl_E2x5max_eg_endcaps;
    MonitorElement *h1_scl_E5x5_eg;
    MonitorElement *h1_scl_E5x5_eg_barrel;
    MonitorElement *h1_scl_E5x5_eg_endcaps;

    MonitorElement *h1_ele_ambiguousTracks;
    MonitorElement *h2_ele_ambiguousTracksVsEta;
    MonitorElement *h2_ele_ambiguousTracksVsPhi;
    MonitorElement *h2_ele_ambiguousTracksVsPt;
    MonitorElement *h1_ele_foundHits;
    MonitorElement *h1_ele_foundHits_barrel;
    MonitorElement *h1_ele_foundHits_endcaps;
    MonitorElement *h2_ele_foundHitsVsEta;
    MonitorElement *h2_ele_foundHitsVsPhi;
    MonitorElement *h2_ele_foundHitsVsPt;
    MonitorElement *h1_ele_lostHits;
    MonitorElement *h1_ele_lostHits_barrel;
    MonitorElement *h1_ele_lostHits_endcaps;
    MonitorElement *h2_ele_lostHitsVsEta;
    MonitorElement *h2_ele_lostHitsVsPhi;
    MonitorElement *h2_ele_lostHitsVsPt;
    MonitorElement *h1_ele_chi2;
    MonitorElement *h1_ele_chi2_barrel;
    MonitorElement *h1_ele_chi2_endcaps;
    MonitorElement *h2_ele_chi2VsEta;
    MonitorElement *h2_ele_chi2VsPhi;
    MonitorElement *h2_ele_chi2VsPt;

    MonitorElement *h1_ele_PoPtrue;
    MonitorElement *h1_ele_PtoPttrue;
    MonitorElement *h2_ele_PoPtrueVsEta;
    MonitorElement *h2_ele_PoPtrueVsPhi;
    MonitorElement *h2_ele_PoPtrueVsPt;
    MonitorElement *h1_ele_PoPtrue_barrel;
    MonitorElement *h1_ele_PoPtrue_endcaps;
    MonitorElement *h1_ele_PoPtrue_golden_barrel;
    MonitorElement *h1_ele_PoPtrue_golden_endcaps;
    MonitorElement *h1_ele_PoPtrue_showering_barrel;
    MonitorElement *h1_ele_PoPtrue_showering_endcaps;
    MonitorElement *h1_ele_PtoPttrue_barrel;
    MonitorElement *h1_ele_PtoPttrue_endcaps;
    MonitorElement *h1_ele_ChargeMnChargeTrue;
    MonitorElement *h1_ele_EtaMnEtaTrue;
    MonitorElement *h1_ele_EtaMnEtaTrue_barrel;
    MonitorElement *h1_ele_EtaMnEtaTrue_endcaps;
    MonitorElement *h2_ele_EtaMnEtaTrueVsEta;
    MonitorElement *h2_ele_EtaMnEtaTrueVsPhi;
    MonitorElement *h2_ele_EtaMnEtaTrueVsPt;
    MonitorElement *h1_ele_PhiMnPhiTrue;
    MonitorElement *h1_ele_PhiMnPhiTrue_barrel;
    MonitorElement *h1_ele_PhiMnPhiTrue_endcaps;
    MonitorElement *h1_ele_PhiMnPhiTrue2;
    MonitorElement *h2_ele_PhiMnPhiTrueVsEta;
    MonitorElement *h2_ele_PhiMnPhiTrueVsPhi;
    MonitorElement *h2_ele_PhiMnPhiTrueVsPt;
    MonitorElement *h1_ele_PinMnPout;
    MonitorElement *h1_ele_PinMnPout_mode;
    MonitorElement *h2_ele_PinMnPoutVsEta_mode;
    MonitorElement *h2_ele_PinMnPoutVsPhi_mode;
    MonitorElement *h2_ele_PinMnPoutVsPt_mode;
    MonitorElement *h2_ele_PinMnPoutVsE_mode;
    MonitorElement *h2_ele_PinMnPoutVsChi2_mode;

    MonitorElement *h1_ele_outerP;
    MonitorElement *h1_ele_outerP_mode;
    MonitorElement *h2_ele_outerPVsEta_mode;
    MonitorElement *h1_ele_outerPt;
    MonitorElement *h1_ele_outerPt_mode;
    MonitorElement *h2_ele_outerPtVsEta_mode;
    MonitorElement *h2_ele_outerPtVsPhi_mode;
    MonitorElement *h2_ele_outerPtVsPt_mode;
    MonitorElement *h1_ele_EoP;
    MonitorElement *h1_ele_EoP_barrel;
    MonitorElement *h1_ele_EoP_endcaps;
    MonitorElement *h1_ele_EoP_eg;
    MonitorElement *h1_ele_EoP_eg_barrel;
    MonitorElement *h1_ele_EoP_eg_endcaps;
    MonitorElement *h2_ele_EoPVsEta;
    MonitorElement *h2_ele_EoPVsPhi;
    MonitorElement *h2_ele_EoPVsE;
    MonitorElement *h1_ele_EseedOP;
    MonitorElement *h1_ele_EseedOP_barrel;
    MonitorElement *h1_ele_EseedOP_endcaps;
    MonitorElement *h1_ele_EseedOP_eg;
    MonitorElement *h1_ele_EseedOP_eg_barrel;
    MonitorElement *h1_ele_EseedOP_eg_endcaps;
    MonitorElement *h2_ele_EseedOPVsEta;
    MonitorElement *h2_ele_EseedOPVsPhi;
    MonitorElement *h2_ele_EseedOPVsE;
    MonitorElement *h1_ele_EoPout;
    MonitorElement *h1_ele_EoPout_barrel;
    MonitorElement *h1_ele_EoPout_endcaps;
    MonitorElement *h1_ele_EoPout_eg;
    MonitorElement *h1_ele_EoPout_eg_barrel;
    MonitorElement *h1_ele_EoPout_eg_endcaps;
    MonitorElement *h2_ele_EoPoutVsEta;
    MonitorElement *h2_ele_EoPoutVsPhi;
    MonitorElement *h2_ele_EoPoutVsE;
    MonitorElement *h1_ele_EeleOPout;
    MonitorElement *h1_ele_EeleOPout_barrel;
    MonitorElement *h1_ele_EeleOPout_endcaps;
    MonitorElement *h1_ele_EeleOPout_eg;
    MonitorElement *h1_ele_EeleOPout_eg_barrel;
    MonitorElement *h1_ele_EeleOPout_eg_endcaps;
    MonitorElement *h2_ele_EeleOPoutVsEta;
    MonitorElement *h2_ele_EeleOPoutVsPhi;
    MonitorElement *h2_ele_EeleOPoutVsE;

    MonitorElement *h1_ele_dEtaSc_propVtx;
    MonitorElement *h1_ele_dEtaSc_propVtx_barrel;
    MonitorElement *h1_ele_dEtaSc_propVtx_endcaps;
    MonitorElement *h1_ele_dEtaSc_propVtx_eg;
    MonitorElement *h1_ele_dEtaSc_propVtx_eg_barrel;
    MonitorElement *h1_ele_dEtaSc_propVtx_eg_endcaps;
    MonitorElement *h2_ele_dEtaScVsEta_propVtx;
    MonitorElement *h2_ele_dEtaScVsPhi_propVtx;
    MonitorElement *h2_ele_dEtaScVsPt_propVtx;
    MonitorElement *h1_ele_dPhiSc_propVtx;
    MonitorElement *h1_ele_dPhiSc_propVtx_barrel;
    MonitorElement *h1_ele_dPhiSc_propVtx_endcaps;
    MonitorElement *h1_ele_dPhiSc_propVtx_eg;
    MonitorElement *h1_ele_dPhiSc_propVtx_eg_barrel;
    MonitorElement *h1_ele_dPhiSc_propVtx_eg_endcaps;
    MonitorElement *h2_ele_dPhiScVsEta_propVtx;
    MonitorElement *h2_ele_dPhiScVsPhi_propVtx;
    MonitorElement *h2_ele_dPhiScVsPt_propVtx;
    MonitorElement *h1_ele_dEtaCl_propOut;
    MonitorElement *h1_ele_dEtaCl_propOut_barrel;
    MonitorElement *h1_ele_dEtaCl_propOut_endcaps;
    MonitorElement *h1_ele_dEtaCl_propOut_eg;
    MonitorElement *h1_ele_dEtaCl_propOut_eg_barrel;
    MonitorElement *h1_ele_dEtaCl_propOut_eg_endcaps;
    MonitorElement *h2_ele_dEtaClVsEta_propOut;
    MonitorElement *h2_ele_dEtaClVsPhi_propOut;
    MonitorElement *h2_ele_dEtaClVsPt_propOut;
    MonitorElement *h1_ele_dPhiCl_propOut;
    MonitorElement *h1_ele_dPhiCl_propOut_barrel;
    MonitorElement *h1_ele_dPhiCl_propOut_endcaps;
    MonitorElement *h1_ele_dPhiCl_propOut_eg;
    MonitorElement *h1_ele_dPhiCl_propOut_eg_barrel;
    MonitorElement *h1_ele_dPhiCl_propOut_eg_endcaps;
    MonitorElement *h2_ele_dPhiClVsEta_propOut;
    MonitorElement *h2_ele_dPhiClVsPhi_propOut;
    MonitorElement *h2_ele_dPhiClVsPt_propOut;
    MonitorElement *h1_ele_dEtaEleCl_propOut;
    MonitorElement *h1_ele_dEtaEleCl_propOut_barrel;
    MonitorElement *h1_ele_dEtaEleCl_propOut_endcaps;
    MonitorElement *h1_ele_dEtaEleCl_propOut_eg;
    MonitorElement *h1_ele_dEtaEleCl_propOut_eg_barrel;
    MonitorElement *h1_ele_dEtaEleCl_propOut_eg_endcaps;
    MonitorElement *h2_ele_dEtaEleClVsEta_propOut;
    MonitorElement *h2_ele_dEtaEleClVsPhi_propOut;
    MonitorElement *h2_ele_dEtaEleClVsPt_propOut;
    MonitorElement *h1_ele_dPhiEleCl_propOut;
    MonitorElement *h1_ele_dPhiEleCl_propOut_barrel;
    MonitorElement *h1_ele_dPhiEleCl_propOut_endcaps;
    MonitorElement *h1_ele_dPhiEleCl_propOut_eg;
    MonitorElement *h1_ele_dPhiEleCl_propOut_eg_barrel;
    MonitorElement *h1_ele_dPhiEleCl_propOut_eg_endcaps;
    MonitorElement *h2_ele_dPhiEleClVsEta_propOut;
    MonitorElement *h2_ele_dPhiEleClVsPhi_propOut;
    MonitorElement *h2_ele_dPhiEleClVsPt_propOut;

    MonitorElement *h1_ele_seed_subdet2;
    MonitorElement *h1_ele_seed_mask;
    MonitorElement *h1_ele_seed_mask_bpix;
    MonitorElement *h1_ele_seed_mask_fpix;
    MonitorElement *h1_ele_seed_mask_tec;
    MonitorElement *h1_ele_seed_dphi2;
    MonitorElement *h2_ele_seed_dphi2VsEta;
    MonitorElement *h2_ele_seed_dphi2VsPt;
    MonitorElement *h1_ele_seed_dphi2pos;
    MonitorElement *h2_ele_seed_dphi2posVsEta;
    MonitorElement *h2_ele_seed_dphi2posVsPt;
    MonitorElement *h1_ele_seed_drz2;
    MonitorElement *h2_ele_seed_drz2VsEta;
    MonitorElement *h2_ele_seed_drz2VsPt;
    MonitorElement *h1_ele_seed_drz2pos;
    MonitorElement *h2_ele_seed_drz2posVsEta;
    MonitorElement *h2_ele_seed_drz2posVsPt;

    MonitorElement *h1_ele_classes;
    MonitorElement *h1_ele_eta;
    MonitorElement *h1_ele_eta_golden;
    MonitorElement *h1_ele_eta_bbrem;
    MonitorElement *h1_ele_eta_shower;

    MonitorElement *h1_ele_HoE;
    MonitorElement *h1_ele_HoE_bc;
    MonitorElement *h1_ele_HoE_barrel;
    MonitorElement *h1_ele_HoE_endcaps;
    MonitorElement *h1_ele_HoE_bc_barrel;
    MonitorElement *h1_ele_HoE_bc_endcaps;
    MonitorElement *h1_ele_HoE_eg;
    MonitorElement *h1_ele_HoE_eg_barrel;
    MonitorElement *h1_ele_HoE_eg_endcaps;
    MonitorElement *h1_ele_HoE_fiducial;
    MonitorElement *h2_ele_HoEVsEta;
    MonitorElement *h2_ele_HoEVsPhi;
    MonitorElement *h2_ele_HoEVsE;

    MonitorElement *h1_ele_fbrem;
    MonitorElement *h1_ele_fbrem_barrel;
    MonitorElement *h1_ele_fbrem_endcaps;
    MonitorElement *h1_ele_fbrem_eg;
    MonitorElement *p1_ele_fbremVsEta_mode;
    MonitorElement *p1_ele_fbremVsEta_mean;

    MonitorElement *h2_ele_PinVsPoutGolden_mode;
    MonitorElement *h2_ele_PinVsPoutShowering_mode;
    MonitorElement *h2_ele_PinVsPoutGolden_mean;
    MonitorElement *h2_ele_PinVsPoutShowering_mean;
    MonitorElement *h2_ele_PtinVsPtoutGolden_mode;
    MonitorElement *h2_ele_PtinVsPtoutShowering_mode;
    MonitorElement *h2_ele_PtinVsPtoutGolden_mean;
    MonitorElement *h2_ele_PtinVsPtoutShowering_mean;
    MonitorElement *h1_scl_EoEtrueGolden_barrel;
    MonitorElement *h1_scl_EoEtrueGolden_endcaps;
    MonitorElement *h1_scl_EoEtrueShowering_barrel;
    MonitorElement *h1_scl_EoEtrueShowering_endcaps;

    MonitorElement *h1_ele_mva;
    MonitorElement *h1_ele_mva_barrel;
    MonitorElement *h1_ele_mva_endcaps;
    MonitorElement *h1_ele_mva_eg;
    MonitorElement *h1_ele_provenance;
    MonitorElement *h1_ele_provenance_barrel;
    MonitorElement *h1_ele_provenance_endcaps;

    // pflow isolation
    MonitorElement *h1_ele_chargedHadronIso;
    MonitorElement *h1_ele_chargedHadronIso_barrel;
    MonitorElement *h1_ele_chargedHadronIso_endcaps;
    MonitorElement *h1_ele_neutralHadronIso;
    MonitorElement *h1_ele_neutralHadronIso_barrel;
    MonitorElement *h1_ele_neutralHadronIso_endcaps;
    MonitorElement *h1_ele_photonIso;
    MonitorElement *h1_ele_photonIso_barrel;
    MonitorElement *h1_ele_photonIso_endcaps;
	// -- pflow over pT
	MonitorElement *h1_ele_chargedHadronRelativeIso;
	MonitorElement *h1_ele_chargedHadronRelativeIso_barrel;
    MonitorElement *h1_ele_chargedHadronRelativeIso_endcaps;
	MonitorElement *h1_ele_neutralHadronRelativeIso;
	MonitorElement *h1_ele_neutralHadronRelativeIso_barrel;
    MonitorElement *h1_ele_neutralHadronRelativeIso_endcaps;
	MonitorElement *h1_ele_photonRelativeIso;
	MonitorElement *h1_ele_photonRelativeIso_barrel;
    MonitorElement *h1_ele_photonRelativeIso_endcaps;

    // isolation
    MonitorElement *h1_ele_tkSumPt_dr03;
    MonitorElement *h1_ele_tkSumPt_dr03_barrel;
    MonitorElement *h1_ele_tkSumPt_dr03_endcaps;
    MonitorElement *h1_ele_ecalRecHitSumEt_dr03;
    MonitorElement *h1_ele_ecalRecHitSumEt_dr03_barrel;
    MonitorElement *h1_ele_ecalRecHitSumEt_dr03_endcaps;
    MonitorElement *h1_ele_hcalTowerSumEt_dr03_depth1;
    MonitorElement *h1_ele_hcalTowerSumEt_dr03_depth1_barrel;
    MonitorElement *h1_ele_hcalTowerSumEt_dr03_depth1_endcaps;
    MonitorElement *h1_ele_hcalTowerSumEt_dr03_depth2;
    MonitorElement *h1_ele_hcalTowerSumEt_dr03_depth2_barrel;
    MonitorElement *h1_ele_hcalTowerSumEt_dr03_depth2_endcaps;
    MonitorElement *h1_ele_tkSumPt_dr04;
    MonitorElement *h1_ele_tkSumPt_dr04_barrel;
    MonitorElement *h1_ele_tkSumPt_dr04_endcaps;
    MonitorElement *h1_ele_ecalRecHitSumEt_dr04;
    MonitorElement *h1_ele_ecalRecHitSumEt_dr04_barrel;
    MonitorElement *h1_ele_ecalRecHitSumEt_dr04_endcaps;
    MonitorElement *h1_ele_hcalTowerSumEt_dr04_depth1;
    MonitorElement *h1_ele_hcalTowerSumEt_dr04_depth1_barrel;
    MonitorElement *h1_ele_hcalTowerSumEt_dr04_depth1_endcaps;
    MonitorElement *h1_ele_hcalTowerSumEt_dr04_depth2;
    MonitorElement *h1_ele_hcalTowerSumEt_dr04_depth2_barrel;
    MonitorElement *h1_ele_hcalTowerSumEt_dr04_depth2_endcaps;

    MonitorElement *h1_ele_dIso_tkSumPt_dr03;
    MonitorElement *h1_ele_dIso_tkSumPt_dr04;
    MonitorElement *h1_ele_dIso_ecalFullRecHitSumEt_dr03;
    MonitorElement *h1_ele_dIso_ecalFullRecHitSumEt_dr04;
    MonitorElement *h1_ele_dIso_ecalReducedRecHitSumEt_dr03;
    MonitorElement *h1_ele_dIso_ecalReducedRecHitSumEt_dr04;
    MonitorElement *h1_ele_dIso_hcalTowerSumEt_dr03;
    MonitorElement *h1_ele_dIso_hcalTowerSumEt_dr04;

    MonitorElement *h1_ele_hcalDepth1OverEcalBc;
    MonitorElement *h1_ele_hcalDepth1OverEcalBc_barrel;
    MonitorElement *h1_ele_hcalDepth1OverEcalBc_endcaps;
    MonitorElement *h1_ele_hcalDepth2OverEcalBc;
    MonitorElement *h1_ele_hcalDepth2OverEcalBc_barrel;
    MonitorElement *h1_ele_hcalDepth2OverEcalBc_endcaps;

    MonitorElement *h1_ele_hcalTowerSumEtBc_dr03_depth1;
    MonitorElement *h1_ele_hcalTowerSumEtBc_dr03_depth1_barrel;
    MonitorElement *h1_ele_hcalTowerSumEtBc_dr03_depth1_endcaps;
    MonitorElement *h1_ele_hcalTowerSumEtBc_dr03_depth2;
    MonitorElement *h1_ele_hcalTowerSumEtBc_dr03_depth2_barrel;
    MonitorElement *h1_ele_hcalTowerSumEtBc_dr03_depth2_endcaps;
    MonitorElement *h1_ele_hcalTowerSumEtBc_dr04_depth1;
    MonitorElement *h1_ele_hcalTowerSumEtBc_dr04_depth1_barrel;
    MonitorElement *h1_ele_hcalTowerSumEtBc_dr04_depth1_endcaps;
    MonitorElement *h1_ele_hcalTowerSumEtBc_dr04_depth2;
    MonitorElement *h1_ele_hcalTowerSumEtBc_dr04_depth2_barrel;
    MonitorElement *h1_ele_hcalTowerSumEtBc_dr04_depth2_endcaps;

    // conversions
    MonitorElement *h1_ele_convFlags;
    MonitorElement *h1_ele_convFlags_all;
    MonitorElement *h1_ele_convDist;
    MonitorElement *h1_ele_convDist_all;
    MonitorElement *h1_ele_convDcot;
    MonitorElement *h1_ele_convDcot_all;
    MonitorElement *h1_ele_convRadius;
    MonitorElement *h1_ele_convRadius_all;

 } ;

#endif



