
// user include files
#include "Validation/RecoEgamma/interface/ElectronMcFakeValidator.h"

#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronUtilities.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/EDProduct.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "TMath.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH1I.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TTree.h"
#include <vector>
#include <iostream>

using namespace reco;

ElectronMcFakeValidator::ElectronMcFakeValidator( const edm::ParameterSet & conf )
 : ElectronValidator(conf)
 {
  outputFile_ = conf.getParameter<std::string>("outputFile");
  electronCollection_ = conf.getParameter<edm::InputTag>("electronCollection");
  matchingObjectCollection_ = conf.getParameter<edm::InputTag>("matchingObjectCollection");
  beamSpotTag_ = conf.getParameter<edm::InputTag>("beamSpot");
  readAOD_ = conf.getParameter<bool>("readAOD");
  maxPt_ = conf.getParameter<double>("MaxPt");
  maxAbsEta_ = conf.getParameter<double>("MaxAbsEta");
  deltaR_ = conf.getParameter<double>("DeltaR");

  // histos bining and limits

  edm::ParameterSet histosSet = conf.getParameter<edm::ParameterSet>("histosCfg") ;

  xyz_nbin=histosSet.getParameter<int>("Nbinxyz");

  p_nbin=histosSet.getParameter<int>("Nbinp");
  p2D_nbin=histosSet.getParameter<int>("Nbinp2D");
  p_max=histosSet.getParameter<double>("Pmax");

  pt_nbin=histosSet.getParameter<int>("Nbinpt");
  pt2D_nbin=histosSet.getParameter<int>("Nbinpt2D");
  pteff_nbin=histosSet.getParameter<int>("Nbinpteff");
  pt_max=histosSet.getParameter<double>("Ptmax");

  fhits_nbin=histosSet.getParameter<int>("Nbinfhits");
  fhits_max=histosSet.getParameter<double>("Fhitsmax");

  lhits_nbin=histosSet.getParameter<int>("Nbinlhits");
  lhits_max=histosSet.getParameter<double>("Lhitsmax");

  eop_nbin=histosSet.getParameter<int>("Nbineop");
  eop2D_nbin=histosSet.getParameter<int>("Nbineop2D");
  eop_max=histosSet.getParameter<double>("Eopmax");
  eopmaxsht=histosSet.getParameter<double>("Eopmaxsht");

  eta_nbin=histosSet.getParameter<int>("Nbineta");
  eta2D_nbin=histosSet.getParameter<int>("Nbineta2D");
  eta_min=histosSet.getParameter<double>("Etamin");
  eta_max=histosSet.getParameter<double>("Etamax");

  deta_nbin=histosSet.getParameter<int>("Nbindeta");
  deta_min=histosSet.getParameter<double>("Detamin");
  deta_max=histosSet.getParameter<double>("Detamax");

  detamatch_nbin=histosSet.getParameter<int>("Nbindetamatch");
  detamatch2D_nbin=histosSet.getParameter<int>("Nbindetamatch2D");
  detamatch_min=histosSet.getParameter<double>("Detamatchmin");
  detamatch_max=histosSet.getParameter<double>("Detamatchmax");

  phi_nbin=histosSet.getParameter<int>("Nbinphi");
  phi2D_nbin=histosSet.getParameter<int>("Nbinphi2D");
  phi_min=histosSet.getParameter<double>("Phimin");
  phi_max=histosSet.getParameter<double>("Phimax");

  dphi_nbin=histosSet.getParameter<int>("Nbindphi");
  dphi_min=histosSet.getParameter<double>("Dphimin");
  dphi_max=histosSet.getParameter<double>("Dphimax");

  dphimatch_nbin=histosSet.getParameter<int>("Nbindphimatch");
  dphimatch2D_nbin=histosSet.getParameter<int>("Nbindphimatch2D");
  dphimatch_min=histosSet.getParameter<double>("Dphimatchmin");
  dphimatch_max=histosSet.getParameter<double>("Dphimatchmax");

  mee_nbin= histosSet.getParameter<int>("Nbinmee");
  mee_min=histosSet.getParameter<double>("Meemin");
  mee_max=histosSet.getParameter<double>("Meemax");

  hoe_nbin= histosSet.getParameter<int>("Nbinhoe");
  hoe_min=histosSet.getParameter<double>("Hoemin");
  hoe_max=histosSet.getParameter<double>("Hoemax");

  popmatching_nbin= histosSet.getParameter<int>("Nbinpopmatching");
  popmatching_min=histosSet.getParameter<double>("Popmatchingmin");
  popmatching_max=histosSet.getParameter<double>("Popmatchingmax");
 }

void ElectronMcFakeValidator::beginJob()
 {
  prepareStore() ;
  setStoreFolder("EgammaV/ElectronMcFakeValidator") ;

  // matching object type
  std::string matchingObjectType ;
  if (std::string::npos!=matchingObjectCollection_.label().find("iterativeCone5GenJets",0))
   { matchingObjectType = "GenJet" ; }
  if (matchingObjectType=="")
   { edm::LogError("ElectronMcFakeValidator::beginJob")<<"Unknown matching object type !" ; }
  else
   { edm::LogInfo("ElectronMcFakeValidator::beginJob")<<"Matching object type: "<<matchingObjectType ; }
  std::string htitle = "# "+matchingObjectType+"s", xtitle = "N_{"+matchingObjectType+"}" ;
  h1_matchingObjectNum = bookH1withSumw2("h_matchingObjectNum",htitle,fhits_nbin,0.,fhits_max,xtitle) ;

  // rec event
  h1_recEleNum_= bookH1("h_recEleNum","# rec electrons",20, 0.,20.,"N_{ele}");

  // mc
  h1_matchingObjectEta = bookH1withSumw2("h_matchingObject_eta",matchingObjectType+" #eta",eta_nbin,eta_min,eta_max,"#eta");
  h1_matchingObjectAbsEta = bookH1withSumw2("h_matchingObject_abseta",matchingObjectType+" |#eta|",eta_nbin/2,0.,eta_max);
  h1_matchingObjectP = bookH1withSumw2("h_matchingObject_P",matchingObjectType+" p",p_nbin,0.,p_max,"p (GeV/c)");
  h1_matchingObjectPt = bookH1withSumw2("h_matchingObject_Pt",matchingObjectType+" pt",pteff_nbin,5.,pt_max);
  h1_matchingObjectPhi = bookH1withSumw2("h_matchingObject_phi",matchingObjectType+" phi",phi_nbin,phi_min,phi_max);
  h1_matchingObjectZ = bookH1withSumw2("h_matchingObject_z",matchingObjectType+" z",xyz_nbin, -25, 25 );

  // all electrons
  h1_ele_EoverP_all = bookH1withSumw2("h_ele_EoverP_all","ele E/P_{vertex}, all reco electrons",eop_nbin,0.,eop_max,"E/P_{vertex}");
  h1_ele_EseedOP_all = bookH1withSumw2("h_ele_EseedOP_all","ele E_{seed}/P_{vertex}, all reco electrons",eop_nbin,0.,eop_max,"E_{seed}/P_{vertex}");
  h1_ele_EoPout_all = bookH1withSumw2("h_ele_EoPout_all","ele E_{seed}/P_{out}, all reco electrons",eop_nbin,0.,eop_max,"E_{seed}/P_{out}");
  h1_ele_EeleOPout_all = bookH1withSumw2("h_ele_EeleOPout_all","ele E_{ele}/P_{out}, all reco electrons",eop_nbin,0.,eop_max,"E_{ele}/P_{out}");
  h1_ele_dEtaSc_propVtx_all = bookH1withSumw2("h_ele_dEtaSc_propVtx_all","ele #eta_{sc} - #eta_{tr}, prop from vertex, all reco electrons",detamatch_nbin,detamatch_min,detamatch_max,"#eta_{sc} - #eta_{tr}");
  h1_ele_dPhiSc_propVtx_all = bookH1withSumw2("h_ele_dPhiSc_propVtx_all","ele #phi_{sc} - #phi_{tr}, prop from vertex, all reco electrons",dphimatch_nbin,dphimatch_min,dphimatch_max,"#phi_{sc} - #phi_{tr} (rad)");
  h1_ele_dEtaCl_propOut_all = bookH1withSumw2("h_ele_dEtaCl_propOut_all","ele #eta_{cl} - #eta_{tr}, prop from outermost, all reco electrons",detamatch_nbin,detamatch_min,detamatch_max,"#eta_{sc} - #eta_{tr}");
  h1_ele_dPhiCl_propOut_all = bookH1withSumw2("h_ele_dPhiCl_propOut_all","ele #phi_{cl} - #phi_{tr}, prop from outermost, all reco electrons",dphimatch_nbin,dphimatch_min,dphimatch_max,"#phi_{sc} - #phi_{tr} (rad)");
  h1_ele_TIP_all = bookH1withSumw2("h_ele_TIP_all","ele vertex transverse radius, all reco electrons",  100,0.,0.2,"r_{T} (cm)");
  h1_ele_HoE_all = bookH1withSumw2("h_ele_HoE_all","ele hadronic energy / em energy, all reco electrons",hoe_nbin, hoe_min, hoe_max,"H/E") ;
  h1_ele_vertexEta_all = bookH1withSumw2("h_ele_vertexEta_all","ele eta, all reco electrons",eta_nbin,eta_min,eta_max);
  h1_ele_vertexPt_all = bookH1withSumw2("h_ele_vertexPt_all","ele p_{T}, all reco electrons",pteff_nbin,5.,pt_max);
  h1_ele_mee_all = bookH1withSumw2("h_ele_mee_all","ele pairs invariant mass, all reco electrons",mee_nbin, mee_min, mee_max,"m_{ee} (GeV/c^{2})" );
  h1_ele_mee_os = bookH1withSumw2("h_ele_mee_os","ele pairs invariant mass, opp. sign",mee_nbin, mee_min, mee_max,"m_{e^{+}e^{-}} (GeV/c^{2})" );

  // duplicates
  h2_ele_E2mnE1vsMee_all = bookH2("h_ele_E2mnE1vsMee_all","E2 - E1 vs ele pairs invariant mass, all electrons",mee_nbin, mee_min, mee_max, 100, -50., 50.,"m_{e^{+}e^{-}} (GeV/c^{2})","E2 - E1 (GeV)");
  h2_ele_E2mnE1vsMee_egeg_all = bookH2("h_ele_E2mnE1vsMee_egeg_all","E2 - E1 vs ele pairs invariant mass, ecal driven pairs, all electrons",mee_nbin, mee_min, mee_max, 100, -50., 50.,"m_{e^{+}e^{-}} (GeV/c^{2})","E2 - E1 (GeV)");

//  // charge ID
//  h1_ele_ChargeMnChargeTrue = bookH1withSumw2("h_ele_ChargeMnChargeTrue","ele charge - gen charge ",5,-1.,4.,"q_{rec} - q_{gen}");
//  h1_ele_matchingObjectEta_matched_qmisid = bookH1withSumw2("h_ele_eta_matched_qmisid","charge misid vs gen eta",eta_nbin,eta_min,eta_max);
//  h1_ele_matchingObjectAbsEta_matched_qmisid = bookH1withSumw2("h_ele_abseta_matched_qmisid","charge misid vs gen |eta|",eta_nbin/2,0.,eta_max);
//  h1_ele_matchingObjectPt_matched_qmisid = bookH1withSumw2("h_ele_Pt_matched_qmisid","charge misid vs gen transverse momentum",pteff_nbin,5.,pt_max);
//  h1_ele_matchingObjectPhi_matched_qmisid = bookH1withSumw2("h_ele_phi_matched_qmisid","charge misid vs gen phi",phi_nbin,phi_min,phi_max);
//  h1_ele_matchingObjectZ_matched_qmisid = bookH1withSumw2("h_ele_z_matched_qmisid","charge misid vs gen z",xyz_nbin, -25, 25 );

  // matched electrons

  htitle = "Efficiency vs matching "+matchingObjectType+" " ;
  h1_ele_matchingObjectEta_matched = bookH1withSumw2("h_ele_matchingObjectEta_matched",htitle+"#eta",eta_nbin,eta_min,eta_max);
  h1_ele_matchingObjectAbsEta_matched = bookH1withSumw2("h_ele_matchingObjectAbsEta_matched",htitle+"|#eta|",eta_nbin/2,0.,2.5);
  h1_ele_matchingObjectPt_matched = bookH1("h_ele_matchingObjectPt_matched",htitle+"p_{T}",pteff_nbin,5.,pt_max);
  h1_ele_matchingObjectPhi_matched = bookH1withSumw2("h_ele_matchingObjectPhi_matched",htitle+"phi",phi_nbin,phi_min,phi_max);
  h1_ele_matchingObjectZ_matched = bookH1withSumw2("h_ele_matchingObjectZ_matched",htitle+"z",xyz_nbin,-25,25);

  h1_ele_charge = bookH1withSumw2("h_ele_charge","ele charge",5,-2.,2.,"charge");
  h2_ele_chargeVsEta = bookH2("h_ele_chargeVsEta","ele charge vs eta",eta2D_nbin,eta_min,eta_max,5,-2.,2.);
  h2_ele_chargeVsPhi = bookH2("h_ele_chargeVsPhi","ele charge vs phi",phi2D_nbin,phi_min,phi_max,5,-2.,2.);
  h2_ele_chargeVsPt = bookH2("h_ele_chargeVsPt","ele charge vs pt",pt_nbin,0.,100.,5,-2.,2.);
  h1_ele_vertexP = bookH1withSumw2("h_ele_vertexP","ele momentum",p_nbin,0.,p_max,"p_{vertex} (GeV/c)");
  h1_ele_vertexPt = bookH1withSumw2("h_ele_vertexPt","ele transverse momentum",pt_nbin,0.,pt_max,"p_{T vertex} (GeV/c)");
  h2_ele_vertexPtVsEta = bookH2("h_ele_vertexPtVsEta","ele transverse momentum vs eta",eta2D_nbin,eta_min,eta_max,pt2D_nbin,0.,pt_max);
  h2_ele_vertexPtVsPhi = bookH2("h_ele_vertexPtVsPhi","ele transverse momentum vs phi",phi2D_nbin,phi_min,phi_max,pt2D_nbin,0.,pt_max);
  h1_ele_vertexEta = bookH1withSumw2("h_ele_vertexEta","ele momentum eta",eta_nbin,eta_min,eta_max,"#eta");
  h2_ele_vertexEtaVsPhi = bookH2("h_ele_vertexEtaVsPhi","ele momentum eta vs phi",eta2D_nbin,eta_min,eta_max,phi2D_nbin,phi_min,phi_max );
  h1_ele_vertexPhi = bookH1withSumw2("h_ele_vertexPhi","ele  momentum #phi",phi_nbin,phi_min,phi_max,"#phi (rad)");
  h1_ele_vertexX = bookH1withSumw2("h_ele_vertexX","ele vertex x",xyz_nbin,-0.6,0.6,"x (cm)" );
  h1_ele_vertexY = bookH1withSumw2("h_ele_vertexY","ele vertex y",xyz_nbin,-0.6,0.6,"y (cm)" );
  h1_ele_vertexZ = bookH1withSumw2("h_ele_vertexZ","ele vertex z",xyz_nbin,-25, 25,"z (cm)" );
  h1_ele_vertexTIP = bookH1withSumw2("h_ele_vertexTIP","ele transverse impact parameter (wrt gen vtx)",90,0.,0.15,"TIP (cm)");
  h2_ele_vertexTIPVsEta = bookH2("h_ele_vertexTIPVsEta","ele transverse impact parameter (wrt gen vtx) vs eta",eta2D_nbin,eta_min,eta_max,45,0.,0.15,"#eta","TIP (cm)");
  h2_ele_vertexTIPVsPhi = bookH2("h_ele_vertexTIPVsPhi","ele transverse impact parameter (wrt gen vtx) vs phi",phi2D_nbin,phi_min,phi_max,45,0.,0.15,"#phi (rad)","TIP (cm)");
  h2_ele_vertexTIPVsPt = bookH2("h_ele_vertexTIPVsPt","ele transverse impact parameter (wrt gen vtx) vs transverse momentum",pt2D_nbin,0.,pt_max,45,0.,0.15,"p_{T} (GeV/c)","TIP (cm)");

  htitle = "Ele / matching "+matchingObjectType+", momemtum" ;
  xtitle = "P / P_{"+matchingObjectType+"}" ;
  h1_ele_PoPmatchingObject = bookH1withSumw2("h_ele_PoPmatchingObject",htitle,popmatching_nbin,popmatching_min,popmatching_max,xtitle);
  h2_ele_PoPmatchingObjectVsEta = bookH2("h_ele_PoPmatchingObjectVsEta",htitle+",vs eta",eta2D_nbin,eta_min,eta_max,50,popmatching_min,popmatching_max);
  h2_ele_PoPmatchingObjectVsPhi = bookH2("h_ele_PoPmatchingObjectVsPhi",htitle+",vs phi",phi2D_nbin,phi_min,phi_max,50,popmatching_min,popmatching_max);
  h2_ele_PoPmatchingObjectVsPt = bookH2("h_ele_PoPmatchingObjectVsPt",htitle+",vs eta",pt2D_nbin,0.,pt_max,50,popmatching_min,popmatching_max);
  h1_ele_PoPmatchingObject_barrel = bookH1withSumw2("h_ele_PoPmatchingObject_barrel",htitle+", barrel",popmatching_nbin,popmatching_min,popmatching_max,xtitle);
  h1_ele_PoPmatchingObject_endcaps = bookH1withSumw2("h_ele_PoPmatchingObject_endcaps",htitle+", endcaps",popmatching_nbin,popmatching_min,popmatching_max,xtitle);
  htitle = "Ele - "+matchingObjectType+", " ;
  xtitle = "#eta - #eta_{"+matchingObjectType+"}" ;
  h1_ele_EtaMnEtamatchingObject = bookH1withSumw2("h_ele_EtamatchingObjectEtaTrue",htitle+"eta",deta_nbin,deta_min,deta_max,xtitle);
  h2_ele_EtaMnEtamatchingObjectVsEta = bookH2("h_ele_EtaMnEtamatchingObjectVsEta",htitle+"eta, vs eta",eta2D_nbin,eta_min,eta_max,deta_nbin/2,deta_min,deta_max);
  h2_ele_EtaMnEtamatchingObjectVsPhi = bookH2("h_ele_EtaMnEtamatchingObjectVsPhi",htitle+"eta, vs phi",phi2D_nbin,phi_min,phi_max,deta_nbin/2,deta_min,deta_max);
  h2_ele_EtaMnEtamatchingObjectVsPt = bookH2("h_ele_EtaMnEtamatchingObjectVsPt",htitle+"eta,, vs pt",pt_nbin,0.,pt_max,deta_nbin/2,deta_min,deta_max);
  xtitle = "#phi - #phi_{"+matchingObjectType+"} (rad)" ;
  h1_ele_PhiMnPhimatchingObject = bookH1withSumw2("h_ele_PhiMnPhimatchingObject",htitle+"phi",dphi_nbin,dphi_min,dphi_max,xtitle);
  h1_ele_PhiMnPhimatchingObject2 = bookH1("h_ele_PhiMnPhimatchingObject2",htitle+"phi",dphimatch2D_nbin,dphimatch_min,dphimatch_max);
  h2_ele_PhiMnPhimatchingObjectVsEta = bookH2("h_ele_PhiMnPhimatchingObjectVsEta",htitle+"phi, vs eta",eta2D_nbin,eta_min,eta_max,dphi_nbin/2,dphi_min,dphi_max);
  h2_ele_PhiMnPhimatchingObjectVsPhi = bookH2("h_ele_PhiMnPhimatchingObjectVsPhi",htitle+"phi, vs phi",phi2D_nbin,phi_min,phi_max,dphi_nbin/2,dphi_min,dphi_max);
  h2_ele_PhiMnPhimatchingObjectVsPt = bookH2("h_ele_PhiMnPhimatchingObjectVsPt",htitle+"phi, vs pt",pt2D_nbin,0.,pt_max,dphi_nbin/2,dphi_min,dphi_max);

  // matched electron, superclusters

  h1_scl_En_ = bookH1withSumw2("h_scl_energy","ele supercluster energy",p_nbin,0.,p_max);
  htitle = "Ele supercluster / "+matchingObjectType+", energy" ;
  xtitle = "E/E_{"+matchingObjectType+"}" ;
  h1_scl_EoEmatchingObject_barrel = bookH1withSumw2("h_scl_EoEmatchingObject_barrel",htitle+", barrel",50,0.2,1.2,xtitle);
  h1_scl_EoEmatchingObject_endcaps = bookH1withSumw2("h_scl_EoEmatchingObject_endcaps",htitle+", endcaps",50,0.2,1.2,xtitle);
  h1_scl_Et_ = bookH1withSumw2("h_scl_et","ele supercluster transverse energy",pt_nbin,0.,pt_max);
  h2_scl_EtVsEta_ = bookH2("h_scl_etVsEta","ele supercluster transverse energy vs eta",eta2D_nbin,eta_min,eta_max,pt_nbin,0.,pt_max);
  h2_scl_EtVsPhi_ = bookH2("h_scl_etVsPhi","ele supercluster transverse energy vs phi",phi2D_nbin,phi_min,phi_max,pt_nbin,0.,pt_max);
  h2_scl_EtaVsPhi_ = bookH2("h_scl_etaVsPhi","ele supercluster eta vs phi",phi2D_nbin,phi_min,phi_max,eta2D_nbin,eta_min,eta_max);
  h1_scl_Eta_ = bookH1withSumw2("h_scl_eta","ele supercluster eta",eta_nbin,eta_min,eta_max);
  h1_scl_Phi_ = bookH1withSumw2("h_scl_phi","ele supercluster phi",phi_nbin,phi_min,phi_max);
  h1_scl_SigIEtaIEta_  = bookH1withSumw2("h_scl_sigietaieta","ele supercluster sigma ieta ieta",100,0.,0.05,"#sigma_{i#eta i#eta}");
  h1_scl_SigIEtaIEta_barrel_  = bookH1withSumw2("h_scl_sigietaieta_barrel","ele supercluster sigma ieta ieta, barrel",100,0.,0.05,"#sigma_{i#eta i#eta}");
  h1_scl_SigIEtaIEta_endcaps_  = bookH1withSumw2("h_scl_sigietaieta_endcaps","ele supercluster sigma ieta ieta, endcaps",100,0.,0.05,"#sigma_{i#eta i#eta}");
  h1_scl_E1x5_  = bookH1withSumw2("h_scl_E1x5","ele supercluster energy in 1x5",p_nbin,0., p_max,"E1x5 (GeV)");
  h1_scl_E1x5_barrel_  = bookH1withSumw2("h_scl_E1x5_barrel","ele supercluster energy in 1x5 barrel",p_nbin,0., p_max,"E1x5 (GeV)");
  h1_scl_E1x5_endcaps_  = bookH1withSumw2("h_scl_E1x5_endcaps","ele supercluster energy in 1x5 endcaps",p_nbin,0., p_max,"E1x5 (GeV)");
  h1_scl_E2x5max_  = bookH1withSumw2("h_scl_E2x5max","ele supercluster energy in 2x5 max",p_nbin,0.,p_max,"E2x5 (GeV)");
  h1_scl_E2x5max_barrel_  = bookH1withSumw2("h_scl_E2x5max_barrel","ele supercluster energy in 2x5 _max barrel",p_nbin,0.,p_max,"E2x5 (GeV)");
  h1_scl_E2x5max_endcaps_  = bookH1withSumw2("h_scl_E2x5max_endcaps","ele supercluster energy in 2x5 _max endcaps",p_nbin,0.,p_max,"E2x5 (GeV)");
  h1_scl_E5x5_  = bookH1withSumw2("h_scl_E5x5","ele supercluster energy in 5x5",p_nbin,0.,p_max,"E5x5 (GeV)");
  h1_scl_E5x5_barrel_  = bookH1withSumw2("h_scl_E5x5_barrel","ele supercluster energy in 5x5 barrel",p_nbin,0.,p_max,"E5x5 (GeV)");
  h1_scl_E5x5_endcaps_  = bookH1withSumw2("h_scl_E5x5_endcaps","ele supercluster energy in 5x5 endcaps",p_nbin,0.,p_max,"E5x5 (GeV)");
//  h2_scl_EoEtruePfVsEg = bookH2("h_scl_EoEtruePfVseg","ele supercluster energy / gen energy pflow vs eg",75,-0.1,1.4, 75, -0.1, 1.4,"E/E_{gen} (e/g)","E/E_{gen} (pflow)") ;

  // matched electron, gsf tracks
  h1_ele_ambiguousTracks = bookH1withSumw2("h_ele_ambiguousTracks","ele # ambiguous tracks",  5,0.,5.,"N_{ambiguous tracks}");
  h2_ele_ambiguousTracksVsEta = bookH2("h_ele_ambiguousTracksVsEta","ele # ambiguous tracks  vs eta",eta2D_nbin,eta_min,eta_max,5,0.,5.);
  h2_ele_ambiguousTracksVsPhi = bookH2("h_ele_ambiguousTracksVsPhi","ele # ambiguous tracks  vs phi",phi2D_nbin,phi_min,phi_max,5,0.,5.);
  h2_ele_ambiguousTracksVsPt = bookH2("h_ele_ambiguousTracksVsPt","ele # ambiguous tracks vs pt",pt2D_nbin,0.,pt_max,5,0.,5.);
  h1_ele_foundHits = bookH1withSumw2("h_ele_foundHits","ele track # found hits",fhits_nbin,0.,fhits_max,"N_{hits}");
  h2_ele_foundHitsVsEta = bookH2("h_ele_foundHitsVsEta","ele track # found hits vs eta",eta2D_nbin,eta_min,eta_max,fhits_nbin,0.,fhits_max);
  h2_ele_foundHitsVsPhi = bookH2("h_ele_foundHitsVsPhi","ele track # found hits vs phi",phi2D_nbin,phi_min,phi_max,fhits_nbin,0.,fhits_max);
  h2_ele_foundHitsVsPt = bookH2("h_ele_foundHitsVsPt","ele track # found hits vs pt",pt2D_nbin,0.,pt_max,fhits_nbin,0.,fhits_max);
  h1_ele_lostHits = bookH1withSumw2("h_ele_lostHits","ele track # lost hits",       5,0.,5.,"N_{lost hits}");
  h2_ele_lostHitsVsEta = bookH2("h_ele_lostHitsVsEta","ele track # lost hits vs eta",eta2D_nbin,eta_min,eta_max,lhits_nbin,0.,lhits_max);
  h2_ele_lostHitsVsPhi = bookH2("h_ele_lostHitsVsPhi","ele track # lost hits vs eta",phi2D_nbin,phi_min,phi_max,lhits_nbin,0.,lhits_max);
  h2_ele_lostHitsVsPt = bookH2("h_ele_lostHitsVsPt","ele track # lost hits vs eta",pt2D_nbin,0.,pt_max,lhits_nbin,0.,lhits_max);
  h1_ele_chi2 = bookH1withSumw2("h_ele_chi2","ele track #chi^{2}",100,0.,15.,"#Chi^{2}");
  h2_ele_chi2VsEta = bookH2("h_ele_chi2VsEta","ele track #chi^{2} vs eta",eta2D_nbin,eta_min,eta_max,50,0.,15.);
  h2_ele_chi2VsPhi = bookH2("h_ele_chi2VsPhi","ele track #chi^{2} vs phi",phi2D_nbin,phi_min,phi_max,50,0.,15.);
  h2_ele_chi2VsPt = bookH2("h_ele_chi2VsPt","ele track #chi^{2} vs pt",pt2D_nbin,0.,pt_max,50,0.,15.);
  h1_ele_PinMnPout = bookH1withSumw2("h_ele_PinMnPout","ele track inner p - outer p, mean of GSF components"   ,p_nbin,0.,200.,"P_{vertex} - P_{out} (GeV/c)");
  h1_ele_PinMnPout_mode = bookH1withSumw2("h_ele_PinMnPout_mode","ele track inner p - outer p, mode of GSF components"   ,p_nbin,0.,100.,"P_{vertex} - P_{out}, mode of GSF components (GeV/c)");
  h2_ele_PinMnPoutVsEta_mode = bookH2("h_ele_PinMnPoutVsEta_mode","ele track inner p - outer p vs eta, mode of GSF components" ,eta2D_nbin, eta_min,eta_max,p2D_nbin,0.,100.);
  h2_ele_PinMnPoutVsPhi_mode = bookH2("h_ele_PinMnPoutVsPhi_mode","ele track inner p - outer p vs phi, mode of GSF components" ,phi2D_nbin, phi_min,phi_max,p2D_nbin,0.,100.);
  h2_ele_PinMnPoutVsPt_mode = bookH2("h_ele_PinMnPoutVsPt_mode","ele track inner p - outer p vs pt, mode of GSF components" ,pt2D_nbin, 0.,pt_max,p2D_nbin,0.,100.);
  h2_ele_PinMnPoutVsE_mode = bookH2("h_ele_PinMnPoutVsE_mode","ele track inner p - outer p vs E, mode of GSF components" ,p2D_nbin, 0.,200.,p2D_nbin,0.,100.);
  h2_ele_PinMnPoutVsChi2_mode = bookH2("h_ele_PinMnPoutVsChi2_mode","ele track inner p - outer p vs track chi2, mode of GSF components" ,50, 0.,20.,p2D_nbin,0.,100.);
  h1_ele_outerP = bookH1withSumw2("h_ele_outerP","ele track outer p, mean of GSF components",p_nbin,0.,p_max,"P_{out} (GeV/c)");
  h1_ele_outerP_mode = bookH1withSumw2("h_ele_outerP_mode","ele track outer p, mode of GSF components",p_nbin,0.,p_max,"P_{out} (GeV/c)");
  h2_ele_outerPVsEta_mode = bookH2("h_ele_outerPVsEta_mode","ele track outer p vs eta mode",eta2D_nbin,eta_min,eta_max,50,0.,p_max);
  h1_ele_outerPt = bookH1withSumw2("h_ele_outerPt","ele track outer p_{T}, mean of GSF components",pt_nbin,0.,pt_max,"P_{T out} (GeV/c)");
  h1_ele_outerPt_mode = bookH1withSumw2("h_ele_outerPt_mode","ele track outer p_{T}, mode of GSF components",pt_nbin,0.,pt_max,"P_{T out} (GeV/c)");
  h2_ele_outerPtVsEta_mode = bookH2("h_ele_outerPtVsEta_mode","ele track outer p_{T} vs eta, mode of GSF components",eta2D_nbin,eta_min,eta_max,pt2D_nbin,0.,pt_max);
  h2_ele_outerPtVsPhi_mode = bookH2("h_ele_outerPtVsPhi_mode","ele track outer p_{T} vs phi, mode of GSF components",phi2D_nbin,phi_min,phi_max,pt2D_nbin,0.,pt_max);
  h2_ele_outerPtVsPt_mode = bookH2("h_ele_outerPtVsPt_mode","ele track outer p_{T} vs pt, mode of GSF components",pt2D_nbin,0.,100.,pt2D_nbin,0.,pt_max);

  // matched electrons, matching
  h1_ele_EoP = bookH1withSumw2("h_ele_EoP","ele E/P_{vertex}",eop_nbin,0.,eop_max,"E/P_{vertex}");
  h1_ele_EoP_eg = bookH1withSumw2("h_ele_EoP_eg","ele E/P_{vertex}, ecal driven",eop_nbin,0.,eop_max);
  h1_ele_EoP_barrel = bookH1withSumw2("h_ele_EoP_barrel","ele E/P_{vertex} barrel",eop_nbin,0.,eop_max,"E/P_{vertex}");
  h1_ele_EoP_eg_barrel = bookH1withSumw2("h_ele_EoP_eg_barrel","ele E/P_{vertex}, ecal driven barrel",eop_nbin,0.,eop_max);
  h1_ele_EoP_endcaps = bookH1withSumw2("h_ele_EoP_endcaps","ele E/P_{vertex} endcaps",eop_nbin,0.,eop_max,"E/P_{vertex}");
  h1_ele_EoP_eg_endcaps = bookH1withSumw2("h_ele_EoP_eg_endcaps","ele E/P_{vertex}, ecal driven endcaps",eop_nbin,0.,eop_max);
  h2_ele_EoPVsEta = bookH2("h_ele_EoPVsEta","ele E/P_{vertex} vs eta",eta2D_nbin,eta_min,eta_max,eop2D_nbin,0.,eopmaxsht);
  h2_ele_EoPVsPhi = bookH2("h_ele_EoPVsPhi","ele E/P_{vertex} vs phi",phi2D_nbin,phi_min,phi_max,eop2D_nbin,0.,eopmaxsht);
  h2_ele_EoPVsE = bookH2("h_ele_EoPVsE","ele E/P_{vertex} vs E",  50,0.,p_max ,50,0.,5.);
  h1_ele_EseedOP = bookH1withSumw2("h_ele_EseedOP","ele E_{seed}/P_{vertex}",eop_nbin,0.,eop_max,"E_{seed}/P_{vertex}");
  h1_ele_EseedOP_eg = bookH1withSumw2("h_ele_EseedOP_eg","ele E_{seed}/P_{vertex}, ecal driven",eop_nbin,0.,eop_max);
  h1_ele_EseedOP_barrel = bookH1withSumw2("h_ele_EseedOP_barrel","ele E_{seed}/P_{vertex} barrel",eop_nbin,0.,eop_max,"E_{seed}/P_{vertex}");
  h1_ele_EseedOP_eg_barrel = bookH1withSumw2("h_ele_EseedOP_eg_barrel","ele E_{seed}/P_{vertex}, ecal driven barrel",eop_nbin,0.,eop_max);
  h1_ele_EseedOP_endcaps = bookH1withSumw2("h_ele_EseedOP_endcaps","ele E_{seed}/P_{vertex} endcaps",eop_nbin,0.,eop_max,"E_{seed}/P_{vertex}");
  h1_ele_EseedOP_eg_endcaps = bookH1withSumw2("h_ele_EseedOP_eg_endcaps","ele E_{seed}/P_{vertex}, ecal driven, endcaps",eop_nbin,0.,eop_max);
  h2_ele_EseedOPVsEta = bookH2("h_ele_EseedOPVsEta","ele E_{seed}/P_{vertex} vs eta",eta2D_nbin,eta_min,eta_max,eop2D_nbin,0.,eopmaxsht);
  h2_ele_EseedOPVsPhi = bookH2("h_ele_EseedOPVsPhi","ele E_{seed}/P_{vertex} vs phi",phi2D_nbin,phi_min,phi_max,eop2D_nbin,0.,eopmaxsht);
  h2_ele_EseedOPVsE = bookH2("h_ele_EseedOPVsE","ele E_{seed}/P_{vertex} vs E",  50,0.,p_max ,50,0.,5.);
  h1_ele_EoPout = bookH1withSumw2("h_ele_EoPout","ele E_{seed}/P_{out}",eop_nbin,0.,eop_max,"E_{seed}/P_{out}");
  h1_ele_EoPout_eg = bookH1withSumw2("h_ele_EoPout_eg","ele E_{seed}/P_{out}, ecal driven",eop_nbin,0.,eop_max);
  h1_ele_EoPout_barrel = bookH1withSumw2("h_ele_EoPout_barrel","ele E_{seed}/P_{out} barrel",eop_nbin,0.,eop_max,"E_{seed}/P_{out}");
  h1_ele_EoPout_eg_barrel = bookH1withSumw2("h_ele_EoPout_eg_barrel","ele E_{seed}/P_{out}, ecal driven, barrel",eop_nbin,0.,eop_max);
  h1_ele_EoPout_endcaps = bookH1withSumw2("h_ele_EoPout_endcaps","ele E_{seed}/P_{out} endcaps",eop_nbin,0.,eop_max,"E_{seed}/P_{out}");
  h1_ele_EoPout_eg_endcaps = bookH1withSumw2("h_ele_EoPout_eg_endcaps","ele E_{seed}/P_{out}, ecal driven, endcaps",eop_nbin,0.,eop_max);
  h2_ele_EoPoutVsEta = bookH2("h_ele_EoPoutVsEta","ele E_{seed}/P_{out} vs eta",eta2D_nbin,eta_min,eta_max,eop2D_nbin,0.,eopmaxsht);
  h2_ele_EoPoutVsPhi = bookH2("h_ele_EoPoutVsPhi","ele E_{seed}/P_{out} vs phi",phi2D_nbin,phi_min,phi_max,eop2D_nbin,0.,eopmaxsht);
  h2_ele_EoPoutVsE = bookH2("h_ele_EoPoutVsE","ele E_{seed}/P_{out} vs E",p2D_nbin,0.,p_max,eop2D_nbin,0.,eopmaxsht);
  h1_ele_EeleOPout = bookH1withSumw2("h_ele_EeleOPout","ele E_{ele}/P_{out}",eop_nbin,0.,eop_max,"E_{ele}/P_{out}");
  h1_ele_EeleOPout_eg = bookH1withSumw2("h_ele_EeleOPout_eg","ele E_{ele}/P_{out}, ecal driven",eop_nbin,0.,eop_max);
  h1_ele_EeleOPout_barrel = bookH1withSumw2("h_ele_EeleOPout_barrel","ele E_{ele}/P_{out} barrel",eop_nbin,0.,eop_max,"E_{ele}/P_{out}");
  h1_ele_EeleOPout_eg_barrel = bookH1withSumw2("h_ele_EeleOPout_eg_barrel","ele E_{ele}/P_{out}, ecal driven, barrel",eop_nbin,0.,eop_max);
  h1_ele_EeleOPout_endcaps = bookH1withSumw2("h_ele_EeleOPout_endcaps","ele E_{ele}/P_{out} endcaps",eop_nbin,0.,eop_max,"E_{ele}/P_{out}");
  h1_ele_EeleOPout_eg_endcaps = bookH1withSumw2("h_ele_EeleOPout_eg_endcaps","ele E_{ele}/P_{out}, ecal driven, endcaps",eop_nbin,0.,eop_max);
  h2_ele_EeleOPoutVsEta = bookH2("h_ele_EeleOPoutVsEta","ele E_{ele}/P_{out} vs eta",eta2D_nbin,eta_min,eta_max,eop2D_nbin,0.,eopmaxsht);
  h2_ele_EeleOPoutVsPhi = bookH2("h_ele_EeleOPoutVsPhi","ele E_{ele}/P_{out} vs phi",phi2D_nbin,phi_min,phi_max,eop2D_nbin,0.,eopmaxsht);
  h2_ele_EeleOPoutVsE = bookH2("h_ele_EeleOPoutVsE","ele E_{ele}/P_{out} vs E",p2D_nbin,0.,p_max,eop2D_nbin,0.,eopmaxsht);
  h1_ele_dEtaSc_propVtx = bookH1withSumw2("h_ele_dEtaSc_propVtx","ele #eta_{sc} - #eta_{tr}, prop from vertex",detamatch_nbin,detamatch_min,detamatch_max,"#eta_{sc} - #eta_{tr}");
  h1_ele_dEtaSc_propVtx_eg = bookH1withSumw2("h_ele_dEtaSc_propVtx_eg","ele #eta_{sc} - #eta_{tr}, prop from vertex, ecal driven",detamatch_nbin,detamatch_min,detamatch_max);
  h1_ele_dEtaSc_propVtx_barrel = bookH1withSumw2("h_ele_dEtaSc_propVtx_barrel","ele #eta_{sc} - #eta_{tr}, prop from vertex, barrel",detamatch_nbin,detamatch_min,detamatch_max,"#eta_{sc} - #eta_{tr}");
  h1_ele_dEtaSc_propVtx_eg_barrel = bookH1withSumw2("h_ele_dEtaSc_propVtx_eg_barrel","ele #eta_{sc} - #eta_{tr}, prop from vertex, ecal driven, barrel",detamatch_nbin,detamatch_min,detamatch_max);
  h1_ele_dEtaSc_propVtx_endcaps = bookH1withSumw2("h_ele_dEtaSc_propVtx_endcaps","ele #eta_{sc} - #eta_{tr}, prop from vertex, endcaps",detamatch_nbin,detamatch_min,detamatch_max,"#eta_{sc} - #eta_{tr}");
  h1_ele_dEtaSc_propVtx_eg_endcaps = bookH1withSumw2("h_ele_dEtaSc_propVtx_eg_endcaps","ele #eta_{sc} - #eta_{tr}, prop from vertex, ecal driven, endcaps",detamatch_nbin,detamatch_min,detamatch_max);
  h2_ele_dEtaScVsEta_propVtx = bookH2("h_ele_dEtaScVsEta_propVtx","ele #eta_{sc} - #eta_{tr} vs eta, prop from vertex",eta2D_nbin,eta_min,eta_max,detamatch2D_nbin,detamatch_min,detamatch_max);
  h2_ele_dEtaScVsPhi_propVtx = bookH2("h_ele_dEtaScVsPhi_propVtx","ele #eta_{sc} - #eta_{tr} vs phi, prop from vertex",phi2D_nbin,phi_min,phi_max,detamatch2D_nbin,detamatch_min,detamatch_max);
  h2_ele_dEtaScVsPt_propVtx = bookH2("h_ele_dEtaScVsPt_propVtx","ele #eta_{sc} - #eta_{tr} vs pt, prop from vertex",pt2D_nbin,0.,pt_max,detamatch2D_nbin,detamatch_min,detamatch_max);
  h1_ele_dPhiSc_propVtx = bookH1withSumw2("h_ele_dPhiSc_propVtx","ele #phi_{sc} - #phi_{tr}, prop from vertex",dphimatch_nbin,dphimatch_min,dphimatch_max,"#phi_{sc} - #phi_{tr} (rad)");
  h1_ele_dPhiSc_propVtx_eg = bookH1withSumw2("h_ele_dPhiSc_propVtx_eg","ele #phi_{sc} - #phi_{tr}, prop from vertex, ecal driven",dphimatch_nbin,dphimatch_min,dphimatch_max);
  h1_ele_dPhiSc_propVtx_barrel = bookH1withSumw2("h_ele_dPhiSc_propVtx_barrel","ele #phi_{sc} - #phi_{tr}, prop from vertex, barrel",dphimatch_nbin,dphimatch_min,dphimatch_max,"#phi_{sc} - #phi_{tr} (rad)");
  h1_ele_dPhiSc_propVtx_eg_barrel = bookH1withSumw2("h_ele_dPhiSc_propVtx_eg_barrel","ele #phi_{sc} - #phi_{tr}, prop from vertex, ecal driven, barrel",dphimatch_nbin,dphimatch_min,dphimatch_max);
  h1_ele_dPhiSc_propVtx_endcaps = bookH1withSumw2("h_ele_dPhiSc_propVtx_endcaps","ele #phi_{sc} - #phi_{tr}, prop from vertex, endcaps",dphimatch_nbin,dphimatch_min,dphimatch_max,"#phi_{sc} - #phi_{tr} (rad)");
  h1_ele_dPhiSc_propVtx_eg_endcaps = bookH1withSumw2("h_ele_dPhiSc_propVtx_eg_endcaps","ele #phi_{sc} - #phi_{tr}, prop from vertex, ecal driven, endcaps",dphimatch_nbin,dphimatch_min,dphimatch_max);
  h2_ele_dPhiScVsEta_propVtx = bookH2("h_ele_dPhiScVsEta_propVtx","ele #phi_{sc} - #phi_{tr} vs eta, prop from vertex",eta2D_nbin,eta_min,eta_max,dphimatch2D_nbin,dphimatch_min,dphimatch_max);
  h2_ele_dPhiScVsPhi_propVtx = bookH2("h_ele_dPhiScVsPhi_propVtx","ele #phi_{sc} - #phi_{tr} vs phi, prop from vertex",phi2D_nbin,phi_min,phi_max,dphimatch2D_nbin,dphimatch_min,dphimatch_max);
  h2_ele_dPhiScVsPt_propVtx = bookH2("h_ele_dPhiScVsPt_propVtx","ele #phi_{sc} - #phi_{tr} vs pt, prop from vertex",pt2D_nbin,0.,pt_max,dphimatch2D_nbin,dphimatch_min,dphimatch_max);
  h1_ele_dEtaCl_propOut = bookH1withSumw2("h_ele_dEtaCl_propOut","ele #eta_{cl} - #eta_{tr}, prop from outermost",detamatch_nbin,detamatch_min,detamatch_max,"#eta_{seedcl} - #eta_{tr}");
  h1_ele_dEtaCl_propOut_eg = bookH1withSumw2("h_ele_dEtaCl_propOut_eg","ele #eta_{cl} - #eta_{tr}, prop from outermost, ecal driven",detamatch_nbin,detamatch_min,detamatch_max);
  h1_ele_dEtaCl_propOut_barrel = bookH1withSumw2("h_ele_dEtaCl_propOut_barrel","ele #eta_{cl} - #eta_{tr}, prop from outermost, barrel",detamatch_nbin,detamatch_min,detamatch_max,"#eta_{seedcl} - #eta_{tr}");
  h1_ele_dEtaCl_propOut_eg_barrel = bookH1withSumw2("h_ele_dEtaCl_propOut_eg_barrel","ele #eta_{cl} - #eta_{tr}, prop from outermost, ecal driven, barrel",detamatch_nbin,detamatch_min,detamatch_max);
  h1_ele_dEtaCl_propOut_endcaps = bookH1withSumw2("h_ele_dEtaCl_propOut_endcaps","ele #eta_{cl} - #eta_{tr}, prop from outermost, endcaps",detamatch_nbin,detamatch_min,detamatch_max,"#eta_{seedcl} - #eta_{tr}");
  h1_ele_dEtaCl_propOut_eg_endcaps = bookH1withSumw2("h_ele_dEtaCl_propOut_eg_endcaps","ele #eta_{cl} - #eta_{tr}, prop from outermost, ecal driven, endcaps",detamatch_nbin,detamatch_min,detamatch_max);
  h2_ele_dEtaClVsEta_propOut = bookH2("h_ele_dEtaClVsEta_propOut","ele #eta_{cl} - #eta_{tr} vs eta, prop from out",eta2D_nbin,eta_min,eta_max,detamatch2D_nbin,detamatch_min,detamatch_max);
  h2_ele_dEtaClVsPhi_propOut = bookH2("h_ele_dEtaClVsPhi_propOut","ele #eta_{cl} - #eta_{tr} vs phi, prop from out",phi2D_nbin,phi_min,phi_max,detamatch2D_nbin,detamatch_min,detamatch_max);
  h2_ele_dEtaClVsPt_propOut = bookH2("h_ele_dEtaScVsPt_propOut","ele #eta_{cl} - #eta_{tr} vs pt, prop from out",pt2D_nbin,0.,pt_max,detamatch2D_nbin,detamatch_min,detamatch_max);
  h1_ele_dPhiCl_propOut = bookH1withSumw2("h_ele_dPhiCl_propOut","ele #phi_{cl} - #phi_{tr}, prop from outermost",dphimatch_nbin,dphimatch_min,dphimatch_max,"#phi_{seedcl} - #phi_{tr} (rad)");
  h1_ele_dPhiCl_propOut_eg = bookH1withSumw2("h_ele_dPhiCl_propOut_eg","ele #phi_{cl} - #phi_{tr}, prop from outermost, ecal driven",dphimatch_nbin,dphimatch_min,dphimatch_max);
  h1_ele_dPhiCl_propOut_barrel = bookH1withSumw2("h_ele_dPhiCl_propOut_barrel","ele #phi_{cl} - #phi_{tr}, prop from outermost, barrel",dphimatch_nbin,dphimatch_min,dphimatch_max,"#phi_{seedcl} - #phi_{tr} (rad)");
  h1_ele_dPhiCl_propOut_eg_barrel = bookH1withSumw2("h_ele_dPhiCl_propOut_eg_barrel","ele #phi_{cl} - #phi_{tr}, prop from outermost, ecal driven, barrel",dphimatch_nbin,dphimatch_min,dphimatch_max);
  h1_ele_dPhiCl_propOut_endcaps = bookH1withSumw2("h_ele_dPhiCl_propOut_endcaps","ele #phi_{cl} - #phi_{tr}, prop from outermost, endcaps",dphimatch_nbin,dphimatch_min,dphimatch_max,"#phi_{seedcl} - #phi_{tr} (rad)");
  h1_ele_dPhiCl_propOut_eg_endcaps = bookH1withSumw2("h_ele_dPhiCl_propOut_eg_endcaps","ele #phi_{cl} - #phi_{tr}, prop from outermost, ecal driven, endcaps",dphimatch_nbin,dphimatch_min,dphimatch_max);
  h2_ele_dPhiClVsEta_propOut = bookH2("h_ele_dPhiClVsEta_propOut","ele #phi_{cl} - #phi_{tr} vs eta, prop from out",eta2D_nbin,eta_min,eta_max,dphimatch2D_nbin,dphimatch_min,dphimatch_max);
  h2_ele_dPhiClVsPhi_propOut = bookH2("h_ele_dPhiClVsPhi_propOut","ele #phi_{cl} - #phi_{tr} vs phi, prop from out",phi2D_nbin,phi_min,phi_max,dphimatch2D_nbin,dphimatch_min,dphimatch_max);
  h2_ele_dPhiClVsPt_propOut = bookH2("h_ele_dPhiSClsPt_propOut","ele #phi_{cl} - #phi_{tr} vs pt, prop from out",pt2D_nbin,0.,pt_max,dphimatch2D_nbin,dphimatch_min,dphimatch_max);
  h1_ele_dEtaEleCl_propOut = bookH1withSumw2("h_ele_dEtaEleCl_propOut","ele #eta_{EleCl} - #eta_{tr}, prop from outermost",detamatch_nbin,detamatch_min,detamatch_max,"#eta_{elecl} - #eta_{tr}");
  h1_ele_dEtaEleCl_propOut_eg = bookH1withSumw2("h_ele_dEtaEleCl_propOut_eg","ele #eta_{EleCl} - #eta_{tr}, prop from outermost, ecal driven",detamatch_nbin,detamatch_min,detamatch_max);
  h1_ele_dEtaEleCl_propOut_barrel = bookH1withSumw2("h_ele_dEtaEleCl_propOut_barrel","ele #eta_{EleCl} - #eta_{tr}, prop from outermost, barrel",detamatch_nbin,detamatch_min,detamatch_max,"#eta_{elecl} - #eta_{tr}");
  h1_ele_dEtaEleCl_propOut_eg_barrel = bookH1withSumw2("h_ele_dEtaEleCl_propOut_eg_barrel","ele #eta_{EleCl} - #eta_{tr}, prop from outermost, ecal driven, barrel",detamatch_nbin,detamatch_min,detamatch_max);
  h1_ele_dEtaEleCl_propOut_endcaps = bookH1withSumw2("h_ele_dEtaEleCl_propOut_endcaps","ele #eta_{EleCl} - #eta_{tr}, prop from outermost, endcaps",detamatch_nbin,detamatch_min,detamatch_max,"#eta_{elecl} - #eta_{tr}");
  h1_ele_dEtaEleCl_propOut_eg_endcaps = bookH1withSumw2("h_ele_dEtaEleCl_propOut_eg_endcaps","ele #eta_{EleCl} - #eta_{tr}, prop from outermost, ecal driven, endcaps",detamatch_nbin,detamatch_min,detamatch_max);
  h2_ele_dEtaEleClVsEta_propOut = bookH2("h_ele_dEtaEleClVsEta_propOut","ele #eta_{EleCl} - #eta_{tr} vs eta, prop from out",eta2D_nbin,eta_min,eta_max,detamatch2D_nbin,detamatch_min,detamatch_max);
  h2_ele_dEtaEleClVsPhi_propOut = bookH2("h_ele_dEtaEleClVsPhi_propOut","ele #eta_{EleCl} - #eta_{tr} vs phi, prop from out",phi2D_nbin,phi_min,phi_max,detamatch2D_nbin,detamatch_min,detamatch_max);
  h2_ele_dEtaEleClVsPt_propOut = bookH2("h_ele_dEtaScVsPt_propOut","ele #eta_{EleCl} - #eta_{tr} vs pt, prop from out",pt2D_nbin,0.,pt_max,detamatch2D_nbin,detamatch_min,detamatch_max);
  h1_ele_dPhiEleCl_propOut = bookH1withSumw2("h_ele_dPhiEleCl_propOut","ele #phi_{EleCl} - #phi_{tr}, prop from outermost",dphimatch_nbin,dphimatch_min,dphimatch_max,"#phi_{elecl} - #phi_{tr} (rad)");
  h1_ele_dPhiEleCl_propOut_eg = bookH1withSumw2("h_ele_dPhiEleCl_propOut_eg","ele #phi_{EleCl} - #phi_{tr}, prop from outermost, ecal driven",dphimatch_nbin,dphimatch_min,dphimatch_max);
  h1_ele_dPhiEleCl_propOut_barrel = bookH1withSumw2("h_ele_dPhiEleCl_propOut_barrel","ele #phi_{EleCl} - #phi_{tr}, prop from outermost, barrel",dphimatch_nbin,dphimatch_min,dphimatch_max,"#phi_{elecl} - #phi_{tr} (rad)");
  h1_ele_dPhiEleCl_propOut_eg_barrel = bookH1withSumw2("h_ele_dPhiEleCl_propOut_eg_barrel","ele #phi_{EleCl} - #phi_{tr}, prop from outermost, ecal driven, barrel",dphimatch_nbin,dphimatch_min,dphimatch_max);
  h1_ele_dPhiEleCl_propOut_endcaps = bookH1withSumw2("h_ele_dPhiEleCl_propOut_endcaps","ele #phi_{EleCl} - #phi_{tr}, prop from outermost, endcaps",dphimatch_nbin,dphimatch_min,dphimatch_max,"#phi_{elecl} - #phi_{tr} (rad)");
  h1_ele_dPhiEleCl_propOut_eg_endcaps = bookH1withSumw2("h_ele_dPhiEleCl_propOut_eg_endcaps","ele #phi_{EleCl} - #phi_{tr}, prop from outermost, ecal driven, endcaps",dphimatch_nbin,dphimatch_min,dphimatch_max);
  h2_ele_dPhiEleClVsEta_propOut = bookH2("h_ele_dPhiEleClVsEta_propOut","ele #phi_{EleCl} - #phi_{tr} vs eta, prop from out",eta2D_nbin,eta_min,eta_max,dphimatch2D_nbin,dphimatch_min,dphimatch_max);
  h2_ele_dPhiEleClVsPhi_propOut = bookH2("h_ele_dPhiEleClVsPhi_propOut","ele #phi_{EleCl} - #phi_{tr} vs phi, prop from out",phi2D_nbin,phi_min,phi_max,dphimatch2D_nbin,dphimatch_min,dphimatch_max);
  h2_ele_dPhiEleClVsPt_propOut = bookH2("h_ele_dPhiSEleClsPt_propOut","ele #phi_{EleCl} - #phi_{tr} vs pt, prop from out",pt2D_nbin,0.,pt_max,dphimatch2D_nbin,dphimatch_min,dphimatch_max);
  h1_ele_HoE = bookH1withSumw2("h_ele_HoE","ele hadronic energy / em energy",hoe_nbin, hoe_min, hoe_max,"H/E") ;
  h1_ele_HoE_eg = bookH1withSumw2("h_ele_HoE_eg","ele hadronic energy / em energy, ecal driven",hoe_nbin, hoe_min, hoe_max) ;
  h1_ele_HoE_barrel = bookH1withSumw2("h_ele_HoE_barrel","ele hadronic energy / em energy, barrel",hoe_nbin, hoe_min, hoe_max,"H/E") ;
  h1_ele_HoE_eg_barrel = bookH1withSumw2("h_ele_HoE_eg_barrel","ele hadronic energy / em energy, ecal driven, barrel",hoe_nbin, hoe_min, hoe_max) ;
  h1_ele_HoE_endcaps = bookH1withSumw2("h_ele_HoE_endcaps","ele hadronic energy / em energy, endcaps",hoe_nbin, hoe_min, hoe_max,"H/E") ;
  h1_ele_HoE_eg_endcaps = bookH1withSumw2("h_ele_HoE_eg_endcaps","ele hadronic energy / em energy, ecal driven, endcaps",hoe_nbin, hoe_min, hoe_max) ;
  h1_ele_HoE_fiducial = bookH1withSumw2("h_ele_HoE_fiducial","ele hadronic energy / em energy, fiducial region",hoe_nbin, hoe_min, hoe_max,"H/E") ;
  h2_ele_HoEVsEta = bookH2("h_ele_HoEVsEta","ele hadronic energy / em energy vs eta",eta_nbin,eta_min,eta_max,hoe_nbin, hoe_min, hoe_max) ;
  h2_ele_HoEVsPhi = bookH2("h_ele_HoEVsPhi","ele hadronic energy / em energy vs phi",phi2D_nbin,phi_min,phi_max,hoe_nbin, hoe_min, hoe_max) ;
  h2_ele_HoEVsE = bookH2("h_ele_HoEVsE","ele hadronic energy / em energy vs E",p_nbin, 0.,300.,hoe_nbin, hoe_min, hoe_max) ;
  h1_ele_seed_dphi2_ = bookH1withSumw2("h_ele_seedDphi2","ele seed dphi 2nd layer", 50,-0.003,+0.003,"#phi_{hit}-#phi_{pred} (rad)") ;
  h2_ele_seed_dphi2VsEta_ = bookH2("h_ele_seedDphi2VsEta","ele seed dphi 2nd layer vs eta",eta2D_nbin,eta_min,eta_max,50,-0.003,+0.003) ;
  h2_ele_seed_dphi2VsPt_ = bookH2("h_ele_seedDphi2VsPt","ele seed dphi 2nd layer vs pt",pt2D_nbin,0.,pt_max,50,-0.003,+0.003) ;
  h1_ele_seed_drz2_ = bookH1withSumw2("h_ele_seedDrz2","ele seed dr (dz) 2nd layer", 50,-0.03,+0.03,"r(z)_{hit}-r(z)_{pred} (cm)") ;
  h2_ele_seed_drz2VsEta_ = bookH2("h_ele_seedDrz2VsEta","ele seed dr/dz 2nd layer vs eta",eta2D_nbin,eta_min,eta_max,50,-0.03,+0.03) ;
  h2_ele_seed_drz2VsPt_ = bookH2("h_ele_seedDrz2VsPt","ele seed dr/dz 2nd layer vs pt",pt2D_nbin,0.,pt_max,50,-0.03,+0.03) ;
  h1_ele_seed_subdet2_ = bookH1withSumw2("h_ele_seedSubdet2","ele seed subdet 2nd layer",10,0.,10.,"2nd hit subdet Id") ;

  // classes
  h1_ele_classes = bookH1withSumw2("h_ele_classes","ele classes",20,0.0,20.,"class Id");
  h1_ele_eta = bookH1withSumw2("h_ele_eta","ele electron eta",eta_nbin/2,0.0,eta_max);
  h1_ele_eta_golden = bookH1withSumw2("h_ele_eta_golden","ele electron eta golden",eta_nbin/2,0.0,eta_max);
  h1_ele_eta_bbrem = bookH1withSumw2("h_ele_eta_bbrem","ele electron eta bbrem",eta_nbin/2,0.0,eta_max);
  h1_ele_eta_narrow = bookH1withSumw2("h_ele_eta_narrow","ele electron eta narrow",eta_nbin/2,0.0,eta_max);
  h1_ele_eta_shower = bookH1withSumw2("h_ele_eta_show","ele electron eta showering",eta_nbin/2,0.0,eta_max);
  h2_ele_PinVsPoutGolden_mode = bookH2("h_ele_PinVsPoutGolden_mode","ele track inner p vs outer p vs eta, golden, mode of GSF components" ,p2D_nbin,0.,p_max,50,0.,p_max);
  h2_ele_PinVsPoutShowering_mode = bookH2("h_ele_PinVsPoutShowering_mode","ele track inner p vs outer p vs eta, showering, mode of GSF components" ,p2D_nbin,0.,p_max,50,0.,p_max);
  h2_ele_PinVsPoutGolden_mean = bookH2("h_ele_PinVsPoutGolden_mean","ele track inner p vs outer p vs eta, golden, mean of GSF components" ,p2D_nbin,0.,p_max,50,0.,p_max);
  h2_ele_PinVsPoutShowering_mean = bookH2("h_ele_PinVsPoutShowering_mean","ele track inner p vs outer p vs eta, showering, mean of GSF components" ,p2D_nbin,0.,p_max,50,0.,p_max);
  h2_ele_PtinVsPtoutGolden_mode = bookH2("h_ele_PtinVsPtoutGolden_mode","ele track inner pt vs outer pt vs eta, golden, mode of GSF components" ,pt2D_nbin,0.,pt_max,50,0.,pt_max);
  h2_ele_PtinVsPtoutShowering_mode = bookH2("h_ele_PtinVsPtoutShowering_mode","ele track inner pt vs outer pt vs eta, showering, mode of GSF components" ,pt2D_nbin,0.,pt_max,50,0.,pt_max);
  h2_ele_PtinVsPtoutGolden_mean = bookH2("h_ele_PtinVsPtoutGolden_mean","ele track inner pt vs outer pt vs eta, golden, mean of GSF components" ,pt2D_nbin,0.,pt_max,50,0.,pt_max);
  h2_ele_PtinVsPtoutShowering_mean = bookH2("h_ele_PtinVsPtoutShowering_mean","ele track inner pt vs outer pt vs eta, showering, mean of GSF components" ,pt2D_nbin,0.,pt_max,50,0.,pt_max);
  h1_scl_EoEmatchingObjectGolden_barrel = bookH1withSumw2("h_scl_EoEmatchingObject_golden_barrel","ele supercluster energy / gen energy, golden, barrel",popmatching_nbin,popmatching_min,popmatching_max);
  h1_scl_EoEmatchingObjectGolden_endcaps = bookH1withSumw2("h_scl_EoEmatchingObject_golden_endcaps","ele supercluster energy / gen energy, golden, endcaps",popmatching_nbin,popmatching_min,popmatching_max);
  h1_scl_EoEmatchingObjectShowering_barrel = bookH1withSumw2("h_scl_EoEmatchingObject_showering_barrel","ele supercluster energy / gen energy, showering, barrel",popmatching_nbin,popmatching_min,popmatching_max);
  h1_scl_EoEmatchingObjectShowering_endcaps = bookH1withSumw2("h_scl_EoEmatchingObject_showering_endcaps","ele supercluster energy / gen energy, showering, endcaps",popmatching_nbin,popmatching_min,popmatching_max);

  // isolation
  h1_ele_tkSumPt_dr03 = bookH1withSumw2("h_ele_tkSumPt_dr03","tk isolation sum, dR=0.3",100,0.0,20.,"TkIsoSum, cone 0.3 (GeV/c)");
  h1_ele_ecalRecHitSumEt_dr03 = bookH1withSumw2("h_ele_ecalRecHitSumEt_dr03","ecal isolation sum, dR=0.3",100,0.0,20.,"EcalIsoSum, cone 0.3 (GeV)");
  h1_ele_hcalTowerSumEt_dr03_depth1 = bookH1withSumw2("h_ele_hcalTowerSumEt_dr03_depth1","hcal depth1 isolation sum, dR=0.3",100,0.0,20.,"Hcal1IsoSum, cone 0.3 (GeV)");
  h1_ele_hcalTowerSumEt_dr03_depth2 = bookH1withSumw2("h_ele_hcalTowerSumEt_dr03_depth2","hcal depth2 isolation sum, dR=0.3",100,0.0,20.,"Hcal2IsoSum, cone 0.3 (GeV)");
  h1_ele_tkSumPt_dr04 = bookH1withSumw2("h_ele_tkSumPt_dr04","tk isolation sum, dR=0.4",100,0.0,20.,"TkIsoSum, cone 0.4 (GeV/c)");
  h1_ele_ecalRecHitSumEt_dr04 = bookH1withSumw2("h_ele_ecalRecHitSumEt_dr04","ecal isolation sum, dR=0.4",100,0.0,20.,"EcalIsoSum, cone 0.4 (GeV)");
  h1_ele_hcalTowerSumEt_dr04_depth1 = bookH1withSumw2("h_ele_hcalTowerSumEt_dr04_depth1","hcal depth1 isolation sum, dR=0.4",100,0.0,20.,"Hcal1IsoSum, cone 0.4 (GeV)");
  h1_ele_hcalTowerSumEt_dr04_depth2 = bookH1withSumw2("h_ele_hcalTowerSumEt_dr04_depth2","hcal depth2 isolation sum, dR=0.4",100,0.0,20.,"Hcal2IsoSum, cone 0.4 (GeV)");

  // fbrem
  h1_ele_fbrem = bookH1withSumw2("h_ele_fbrem","ele brem fraction, mode of GSF components",100,0.,1.,"P_{in} - P_{out} / P_{in}");
  p1_ele_fbremVsEta_mode  = bookP1("h_ele_fbremvsEtamode","mean ele brem fraction vs eta, mode of GSF components",eta2D_nbin,eta_min,eta_max,0.,1.,"#eta","<P_{in} - P_{out} / P_{in}>");
  p1_ele_fbremVsEta_mean  = bookP1("h_ele_fbremvsEtamean","mean ele brem fraction vs eta, mean of GSF components",eta2D_nbin,eta_min,eta_max,0.,1.,"#eta","<P_{in} - P_{out} / P_{in}>");

  // e/g et pflow electrons
  h1_ele_mva = bookH1withSumw2("h_ele_mva","ele identification mva",100,-1.,1.);
  h1_ele_provenance = bookH1withSumw2("h_ele_provenance","ele provenance",5,-2.,3.);
 }

ElectronMcFakeValidator::~ElectronMcFakeValidator()
 {}


//=========================================================================
// Main methods
//=========================================================================

void ElectronMcFakeValidator::analyze( const edm::Event & iEvent, const edm::EventSetup & iSetup )
 {
  // get reco electrons
  edm::Handle<reco::GsfElectronCollection> gsfElectrons;
  iEvent.getByLabel(electronCollection_,gsfElectrons);
  edm::LogInfo("ElectronMcFakeValidator::analyze")
    <<"Treating event "<<iEvent.id()
    <<" with "<<gsfElectrons.product()->size()<<" electrons" ;

  // get gen jets
  edm::Handle<reco::GenJetCollection> genJets ;
  iEvent.getByLabel(matchingObjectCollection_,genJets);

  // get the beamspot from the Event:
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  iEvent.getByLabel(beamSpotTag_,recoBeamSpotHandle);
  const BeamSpot bs = *recoBeamSpotHandle;

  h1_recEleNum_->Fill((*gsfElectrons).size());

  // all rec electrons
  reco::GsfElectronCollection::const_iterator gsfIter ;
  for
   ( gsfIter=gsfElectrons->begin() ;
     gsfIter!=gsfElectrons->end() ;
     gsfIter++ )
   {
    // preselect electrons
    if (gsfIter->pt()>maxPt_ || std::abs(gsfIter->eta())>maxAbsEta_)
     { continue ; }
    h1_ele_EoverP_all->Fill( gsfIter->eSuperClusterOverP() );
    h1_ele_EseedOP_all->Fill( gsfIter->eSeedClusterOverP() );
    h1_ele_EoPout_all->Fill( gsfIter->eSeedClusterOverPout() );
    h1_ele_EeleOPout_all->Fill( gsfIter->eEleClusterOverPout() );
    h1_ele_dEtaSc_propVtx_all->Fill(gsfIter->deltaEtaSuperClusterTrackAtVtx());
    h1_ele_dPhiSc_propVtx_all->Fill(gsfIter->deltaPhiSuperClusterTrackAtVtx());
    h1_ele_dEtaCl_propOut_all->Fill(gsfIter->deltaEtaSeedClusterTrackAtCalo());
    h1_ele_dPhiCl_propOut_all->Fill(gsfIter->deltaPhiSeedClusterTrackAtCalo());
    h1_ele_HoE_all->Fill( gsfIter->hadronicOverEm() );
    double d
     = gsfIter->vertex().x()*gsfIter->vertex().x()
     + gsfIter->vertex().y()*gsfIter->vertex().y() ;
    h1_ele_TIP_all->Fill( sqrt(d) );
    h1_ele_vertexEta_all->Fill( gsfIter->eta() );
    h1_ele_vertexPt_all->Fill( gsfIter->pt() );
    float enrj1=gsfIter->superCluster()->energy();
    // mee
    reco::GsfElectronCollection::const_iterator gsfIter2 ;
    for
     ( gsfIter2=gsfIter+1 ;
       gsfIter2!=gsfElectrons->end() ;
       gsfIter2++ )
     {
      math::XYZTLorentzVector p12 = (*gsfIter).p4()+(*gsfIter2).p4();
      float mee2 = p12.Dot(p12) ;
      h1_ele_mee_all->Fill(sqrt(mee2)) ;
      float enrj2=gsfIter2->superCluster()->energy();
      h2_ele_E2mnE1vsMee_all->Fill(sqrt(mee2),enrj2-enrj1);
      if (gsfIter->ecalDrivenSeed() && gsfIter2->ecalDrivenSeed())
       { h2_ele_E2mnE1vsMee_egeg_all->Fill(sqrt(mee2),enrj2-enrj1) ; }
      if (gsfIter->charge()*gsfIter2->charge()<0.)
       { h1_ele_mee_os->Fill(sqrt(mee2)) ; }
     }
   }

  // association matching object-reco electrons
  int matchingObjectNum=0 ;
  reco::GenJetCollection::const_iterator moIter ;
  for
   ( moIter = genJets->begin() ;
     moIter != genJets->end() ; ++moIter )
   {

    // number of matching objects
    matchingObjectNum++ ;

    if (moIter->energy()/cosh(moIter->eta())> maxPt_ || std::abs(moIter->eta())> maxAbsEta_)
     { continue ; }

    // suppress the endcaps
    //if (std::abs(moIter->eta()) > 1.5) continue;
    // select central z
    //if ( std::abs((*mcIter)->production_vertex()->position().z())>50.) continue;

    h1_matchingObjectEta->Fill( moIter->eta() );
    h1_matchingObjectAbsEta->Fill( std::abs(moIter->eta()) );
    h1_matchingObjectP->Fill( moIter->energy() );
    h1_matchingObjectPt->Fill( moIter->energy()/cosh(moIter->eta()) );
    h1_matchingObjectPhi->Fill( moIter->phi() );
    h1_matchingObjectZ->Fill(  moIter->vz() );

    // looking for the best matching gsf electron
    bool okGsfFound = false;
    double gsfOkRatio = 999999.;

    // find best matched electron
    reco::GsfElectron bestGsfElectron ;
    reco::GsfElectronCollection::const_iterator gsfIter ;
    for
     ( gsfIter=gsfElectrons->begin() ;
       gsfIter!=gsfElectrons->end() ; gsfIter++ )
     {
      double dphi = gsfIter->phi()-moIter->phi();
      if (std::abs(dphi)>CLHEP::pi)
       { dphi = dphi < 0? (CLHEP::twopi) + dphi : dphi - CLHEP::twopi; }
      double deltaR = sqrt(pow((gsfIter->eta()-moIter->eta()),2) + pow(dphi,2));
      if ( deltaR < deltaR_ )
       {
        //if ( (genPc->pdg_id() == 11) && (gsfIter->charge() < 0.) || (genPc->pdg_id() == -11) &&
        //(gsfIter->charge() > 0.) ){
        double tmpGsfRatio = gsfIter->p()/moIter->energy() ;
        if ( std::abs(tmpGsfRatio-1) < std::abs(gsfOkRatio-1) )
         {
          gsfOkRatio = tmpGsfRatio ;
          bestGsfElectron =* gsfIter ;
          okGsfFound = true ;
         }
        //}
       }
     } // loop over rec ele to look for the best one

    // analysis when the matching object is matched by a rec electron
    if (okGsfFound)
     {
      // electron related distributions
      h1_ele_charge->Fill( bestGsfElectron.charge() );
      h2_ele_chargeVsEta->Fill( bestGsfElectron.eta(),bestGsfElectron.charge() );
      h2_ele_chargeVsPhi->Fill( bestGsfElectron.phi(),bestGsfElectron.charge() );
      h2_ele_chargeVsPt->Fill( bestGsfElectron.pt(),bestGsfElectron.charge() );
      h1_ele_vertexP->Fill( bestGsfElectron.p() );
      h1_ele_vertexPt->Fill( bestGsfElectron.pt() );
      h2_ele_vertexPtVsEta->Fill(  bestGsfElectron.eta(),bestGsfElectron.pt() );
      h2_ele_vertexPtVsPhi->Fill(  bestGsfElectron.phi(),bestGsfElectron.pt() );
      h1_ele_vertexEta->Fill( bestGsfElectron.eta() );
      // generated distributions for matched electrons
      h1_ele_matchingObjectPt_matched->Fill( moIter->energy()/cosh(moIter->eta()) );
      h1_ele_matchingObjectPhi_matched->Fill( moIter->phi() );
      h1_ele_matchingObjectAbsEta_matched->Fill( std::abs(moIter->eta()) );
      h1_ele_matchingObjectEta_matched->Fill( moIter->eta() );
      h2_ele_vertexEtaVsPhi->Fill(  bestGsfElectron.phi(),bestGsfElectron.eta() );
      h1_ele_vertexPhi->Fill( bestGsfElectron.phi() );
      h1_ele_vertexX->Fill( bestGsfElectron.vertex().x() );
      h1_ele_vertexY->Fill( bestGsfElectron.vertex().y() );
      h1_ele_vertexZ->Fill( bestGsfElectron.vertex().z() );
      h1_ele_matchingObjectZ_matched->Fill( moIter->vz() );
      double d
       =   (bestGsfElectron.vertex().x()-bs.position().x())
         * (bestGsfElectron.vertex().x()-bs.position().x())
       +   (bestGsfElectron.vertex().y()-bs.position().y())
         *(bestGsfElectron.vertex().y()-bs.position().y()) ;
      d = sqrt(d) ;
      h1_ele_vertexTIP->Fill( d );
      h2_ele_vertexTIPVsEta->Fill(  bestGsfElectron.eta(), d );
      h2_ele_vertexTIPVsPhi->Fill(  bestGsfElectron.phi(), d );
      h2_ele_vertexTIPVsPt->Fill(  bestGsfElectron.pt(), d );
      h1_ele_EtaMnEtamatchingObject->Fill( bestGsfElectron.eta()-moIter->eta());
      h2_ele_EtaMnEtamatchingObjectVsEta->Fill( bestGsfElectron.eta(), bestGsfElectron.eta()-moIter->eta());
      h2_ele_EtaMnEtamatchingObjectVsPhi->Fill( bestGsfElectron.phi(), bestGsfElectron.eta()-moIter->eta());
      h2_ele_EtaMnEtamatchingObjectVsPt->Fill( bestGsfElectron.pt(), bestGsfElectron.eta()-moIter->eta());
      h1_ele_PhiMnPhimatchingObject->Fill( bestGsfElectron.phi()-moIter->phi());
      h1_ele_PhiMnPhimatchingObject2->Fill( bestGsfElectron.phi()-moIter->phi());
      h2_ele_PhiMnPhimatchingObjectVsEta->Fill( bestGsfElectron.eta(), bestGsfElectron.phi()-moIter->phi());
      h2_ele_PhiMnPhimatchingObjectVsPhi->Fill( bestGsfElectron.phi(), bestGsfElectron.phi()-moIter->phi());
      h2_ele_PhiMnPhimatchingObjectVsPt->Fill( bestGsfElectron.pt(), bestGsfElectron.phi()-moIter->phi());
      h1_ele_PoPmatchingObject->Fill( bestGsfElectron.p()/moIter->energy());
      h2_ele_PoPmatchingObjectVsEta->Fill( bestGsfElectron.eta(), bestGsfElectron.p()/moIter->energy());
      h2_ele_PoPmatchingObjectVsPhi->Fill( bestGsfElectron.phi(), bestGsfElectron.p()/moIter->energy());
      h2_ele_PoPmatchingObjectVsPt->Fill( bestGsfElectron.py(), bestGsfElectron.p()/moIter->energy());
      if (bestGsfElectron.isEB()) h1_ele_PoPmatchingObject_barrel->Fill( bestGsfElectron.p()/moIter->energy()) ;
      if (bestGsfElectron.isEE()) h1_ele_PoPmatchingObject_endcaps->Fill( bestGsfElectron.p()/moIter->energy()) ;

      // supercluster related distributions
      reco::SuperClusterRef sclRef = bestGsfElectron.superCluster();
      if (!bestGsfElectron.ecalDrivenSeed()&&bestGsfElectron.trackerDrivenSeed()) sclRef = bestGsfElectron.pflowSuperCluster();
      h1_scl_En_->Fill(sclRef->energy());
      double R=TMath::Sqrt(sclRef->x()*sclRef->x() + sclRef->y()*sclRef->y() +sclRef->z()*sclRef->z());
      double Rt=TMath::Sqrt(sclRef->x()*sclRef->x() + sclRef->y()*sclRef->y());
      h1_scl_Et_->Fill(sclRef->energy()*(Rt/R));
      h2_scl_EtVsEta_->Fill(sclRef->eta(),sclRef->energy()*(Rt/R));
      h2_scl_EtVsPhi_->Fill(sclRef->phi(),sclRef->energy()*(Rt/R));
      if (bestGsfElectron.classification() < 100)  h1_scl_EoEmatchingObject_barrel->Fill(sclRef->energy()/moIter->energy());
      if (bestGsfElectron.classification() >= 100)  h1_scl_EoEmatchingObject_endcaps->Fill(sclRef->energy()/moIter->energy());
      h1_scl_Eta_->Fill(sclRef->eta());
      h2_scl_EtaVsPhi_->Fill(sclRef->phi(),sclRef->eta());
      h1_scl_Phi_->Fill(sclRef->phi());
      h1_scl_SigIEtaIEta_->Fill(bestGsfElectron.scSigmaIEtaIEta());
      if (bestGsfElectron.isEB()) h1_scl_SigIEtaIEta_barrel_->Fill(bestGsfElectron.scSigmaIEtaIEta());
      if (bestGsfElectron.isEE()) h1_scl_SigIEtaIEta_endcaps_->Fill(bestGsfElectron.scSigmaIEtaIEta());
      h1_scl_E1x5_->Fill(bestGsfElectron.scE1x5());
      if (bestGsfElectron.isEB()) h1_scl_E1x5_barrel_->Fill(bestGsfElectron.scE1x5());
      if (bestGsfElectron.isEE()) h1_scl_E1x5_endcaps_->Fill(bestGsfElectron.scE1x5());
      h1_scl_E2x5max_->Fill(bestGsfElectron.scE2x5Max());
      if (bestGsfElectron.isEB()) h1_scl_E2x5max_barrel_->Fill(bestGsfElectron.scE2x5Max());
      if (bestGsfElectron.isEE()) h1_scl_E2x5max_endcaps_->Fill(bestGsfElectron.scE2x5Max());
      h1_scl_E5x5_->Fill(bestGsfElectron.scE5x5());
      if (bestGsfElectron.isEB()) h1_scl_E5x5_barrel_->Fill(bestGsfElectron.scE5x5());
      if (bestGsfElectron.isEE()) h1_scl_E5x5_endcaps_->Fill(bestGsfElectron.scE5x5());

      // track related distributions
      h1_ele_ambiguousTracks->Fill( bestGsfElectron.ambiguousGsfTracksSize() );
      h2_ele_ambiguousTracksVsEta->Fill( bestGsfElectron.eta(), bestGsfElectron.ambiguousGsfTracksSize() );
      h2_ele_ambiguousTracksVsPhi->Fill( bestGsfElectron.phi(), bestGsfElectron.ambiguousGsfTracksSize() );
      h2_ele_ambiguousTracksVsPt->Fill( bestGsfElectron.pt(), bestGsfElectron.ambiguousGsfTracksSize() );
      if (!readAOD_)
       { // track extra does not exist in AOD
        h1_ele_foundHits->Fill( bestGsfElectron.gsfTrack()->numberOfValidHits() );
        h2_ele_foundHitsVsEta->Fill( bestGsfElectron.eta(), bestGsfElectron.gsfTrack()->numberOfValidHits() );
        h2_ele_foundHitsVsPhi->Fill( bestGsfElectron.phi(), bestGsfElectron.gsfTrack()->numberOfValidHits() );
        h2_ele_foundHitsVsPt->Fill( bestGsfElectron.pt(), bestGsfElectron.gsfTrack()->numberOfValidHits() );
        h1_ele_lostHits->Fill( bestGsfElectron.gsfTrack()->numberOfLostHits() );
        h2_ele_lostHitsVsEta->Fill( bestGsfElectron.eta(), bestGsfElectron.gsfTrack()->numberOfLostHits() );
        h2_ele_lostHitsVsPhi->Fill( bestGsfElectron.phi(), bestGsfElectron.gsfTrack()->numberOfLostHits() );
        h2_ele_lostHitsVsPt->Fill( bestGsfElectron.pt(), bestGsfElectron.gsfTrack()->numberOfLostHits() );
        h1_ele_chi2->Fill( bestGsfElectron.gsfTrack()->normalizedChi2() );
        h2_ele_chi2VsEta->Fill( bestGsfElectron.eta(), bestGsfElectron.gsfTrack()->normalizedChi2() );
        h2_ele_chi2VsPhi->Fill( bestGsfElectron.phi(), bestGsfElectron.gsfTrack()->normalizedChi2() );
        h2_ele_chi2VsPt->Fill( bestGsfElectron.pt(), bestGsfElectron.gsfTrack()->normalizedChi2() );
       }
      // from gsf track interface, hence using mean
      if (!readAOD_)
       { // track extra does not exist in AOD
        h1_ele_PinMnPout->Fill( bestGsfElectron.gsfTrack()->innerMomentum().R() - bestGsfElectron.gsfTrack()->outerMomentum().R() );
        h1_ele_outerP->Fill( bestGsfElectron.gsfTrack()->outerMomentum().R() );
        h1_ele_outerPt->Fill( bestGsfElectron.gsfTrack()->outerMomentum().Rho() );
       }
      // from electron interface, hence using mode
      h1_ele_PinMnPout_mode->Fill( bestGsfElectron.trackMomentumAtVtx().R() - bestGsfElectron.trackMomentumOut().R() );
      h2_ele_PinMnPoutVsEta_mode->Fill(  bestGsfElectron.eta(), bestGsfElectron.trackMomentumAtVtx().R() - bestGsfElectron.trackMomentumOut().R() );
      h2_ele_PinMnPoutVsPhi_mode->Fill(  bestGsfElectron.phi(), bestGsfElectron.trackMomentumAtVtx().R() - bestGsfElectron.trackMomentumOut().R() );
      h2_ele_PinMnPoutVsPt_mode->Fill(  bestGsfElectron.pt(), bestGsfElectron.trackMomentumAtVtx().R() - bestGsfElectron.trackMomentumOut().R() );
      h2_ele_PinMnPoutVsE_mode->Fill(  bestGsfElectron.caloEnergy(), bestGsfElectron.trackMomentumAtVtx().R() - bestGsfElectron.trackMomentumOut().R() );
      if (!readAOD_)  // track extra does not exist in AOD
       h2_ele_PinMnPoutVsChi2_mode->Fill
        ( bestGsfElectron.gsfTrack()->normalizedChi2(),
          bestGsfElectron.trackMomentumAtVtx().R() - bestGsfElectron.trackMomentumOut().R() ) ;
      h1_ele_outerP_mode->Fill( bestGsfElectron.trackMomentumOut().R() );
      h2_ele_outerPVsEta_mode->Fill(bestGsfElectron.eta(),  bestGsfElectron.trackMomentumOut().R() );
      h1_ele_outerPt_mode->Fill( bestGsfElectron.trackMomentumOut().Rho() );
      h2_ele_outerPtVsEta_mode->Fill(bestGsfElectron.eta(),  bestGsfElectron.trackMomentumOut().Rho() );
      h2_ele_outerPtVsPhi_mode->Fill(bestGsfElectron.phi(),  bestGsfElectron.trackMomentumOut().Rho() );
      h2_ele_outerPtVsPt_mode->Fill(bestGsfElectron.pt(),  bestGsfElectron.trackMomentumOut().Rho() );

      if (!readAOD_)
       { // track extra does not exist in AOD
        edm::RefToBase<TrajectorySeed> seed = bestGsfElectron.gsfTrack()->extra()->seedRef();
        ElectronSeedRef elseed=seed.castTo<ElectronSeedRef>();
        h1_ele_seed_dphi2_->Fill(elseed->dPhi2());
        h2_ele_seed_dphi2VsEta_->Fill(bestGsfElectron.eta(), elseed->dPhi2());
        h2_ele_seed_dphi2VsPt_->Fill(bestGsfElectron.pt(), elseed->dPhi2()) ;
        h1_ele_seed_drz2_->Fill(elseed->dRz2());
        h2_ele_seed_drz2VsEta_->Fill(bestGsfElectron.eta(), elseed->dRz2());
        h2_ele_seed_drz2VsPt_->Fill(bestGsfElectron.pt(), elseed->dRz2());
        h1_ele_seed_subdet2_->Fill(elseed->subDet2());
       }
      // match distributions
      h1_ele_EoP->Fill( bestGsfElectron.eSuperClusterOverP() );
      if (bestGsfElectron.ecalDrivenSeed()) h1_ele_EoP_eg->Fill( bestGsfElectron.eSuperClusterOverP() );
      if (bestGsfElectron.isEB()) h1_ele_EoP_barrel->Fill( bestGsfElectron.eSuperClusterOverP() );
      if (bestGsfElectron.isEB()&&bestGsfElectron.ecalDrivenSeed()) h1_ele_EoP_eg_barrel->Fill( bestGsfElectron.eSuperClusterOverP() );
      if (bestGsfElectron.isEE()) h1_ele_EoP_endcaps->Fill( bestGsfElectron.eSuperClusterOverP() );
      if (bestGsfElectron.isEE()&&bestGsfElectron.ecalDrivenSeed()) h1_ele_EoP_eg_endcaps->Fill( bestGsfElectron.eSuperClusterOverP() );
      h2_ele_EoPVsEta->Fill(bestGsfElectron.eta(),  bestGsfElectron.eSuperClusterOverP() );
      h2_ele_EoPVsPhi->Fill(bestGsfElectron.phi(),  bestGsfElectron.eSuperClusterOverP() );
      h2_ele_EoPVsE->Fill(bestGsfElectron.caloEnergy(),  bestGsfElectron.eSuperClusterOverP() );
      h1_ele_EseedOP->Fill( bestGsfElectron.eSeedClusterOverP() );
      if (bestGsfElectron.ecalDrivenSeed()) h1_ele_EseedOP_eg->Fill( bestGsfElectron.eSeedClusterOverP() );
      if (bestGsfElectron.isEB()) h1_ele_EseedOP_barrel->Fill( bestGsfElectron.eSeedClusterOverP() );
      if (bestGsfElectron.isEB()&&bestGsfElectron.ecalDrivenSeed()) h1_ele_EseedOP_eg_barrel->Fill( bestGsfElectron.eSeedClusterOverP() );
      if (bestGsfElectron.isEE()) h1_ele_EseedOP_endcaps->Fill( bestGsfElectron.eSeedClusterOverP() );
      if (bestGsfElectron.isEE()&&bestGsfElectron.ecalDrivenSeed()) h1_ele_EseedOP_eg_endcaps->Fill( bestGsfElectron.eSeedClusterOverP() );
      h2_ele_EseedOPVsEta->Fill(bestGsfElectron.eta(),  bestGsfElectron.eSeedClusterOverP() );
      h2_ele_EseedOPVsPhi->Fill(bestGsfElectron.phi(),  bestGsfElectron.eSeedClusterOverP() );
      h2_ele_EseedOPVsE->Fill(bestGsfElectron.caloEnergy(),  bestGsfElectron.eSeedClusterOverP() );
      h1_ele_EoPout->Fill( bestGsfElectron.eSeedClusterOverPout() );
      if (bestGsfElectron.ecalDrivenSeed()) h1_ele_EoPout_eg->Fill( bestGsfElectron.eSeedClusterOverPout() );
      if (bestGsfElectron.isEB()) h1_ele_EoPout_barrel->Fill( bestGsfElectron.eSeedClusterOverPout() );
      if (bestGsfElectron.isEB()&&bestGsfElectron.ecalDrivenSeed()) h1_ele_EoPout_eg_barrel->Fill( bestGsfElectron.eSeedClusterOverPout() );
      if (bestGsfElectron.isEE()) h1_ele_EoPout_endcaps->Fill( bestGsfElectron.eSeedClusterOverPout() );
      if (bestGsfElectron.isEE()&&bestGsfElectron.ecalDrivenSeed()) h1_ele_EoPout_eg_endcaps->Fill( bestGsfElectron.eSeedClusterOverPout() );
      h2_ele_EoPoutVsEta->Fill( bestGsfElectron.eta(), bestGsfElectron.eSeedClusterOverPout() );
      h2_ele_EoPoutVsPhi->Fill( bestGsfElectron.phi(), bestGsfElectron.eSeedClusterOverPout() );
      h2_ele_EoPoutVsE->Fill( bestGsfElectron.caloEnergy(), bestGsfElectron.eSeedClusterOverPout() );
      h1_ele_EeleOPout->Fill( bestGsfElectron.eEleClusterOverPout() );
      if (bestGsfElectron.ecalDrivenSeed()) h1_ele_EeleOPout_eg->Fill( bestGsfElectron.eEleClusterOverPout() );
      if (bestGsfElectron.isEB()) h1_ele_EeleOPout_barrel->Fill( bestGsfElectron.eEleClusterOverPout() );
      if (bestGsfElectron.isEB()&&bestGsfElectron.ecalDrivenSeed()) h1_ele_EeleOPout_eg_barrel->Fill( bestGsfElectron.eEleClusterOverPout() );
      if (bestGsfElectron.isEE()) h1_ele_EeleOPout_endcaps->Fill( bestGsfElectron.eEleClusterOverPout() );
      if (bestGsfElectron.isEE()&&bestGsfElectron.ecalDrivenSeed()) h1_ele_EeleOPout_eg_endcaps->Fill( bestGsfElectron.eEleClusterOverPout() );
      h2_ele_EeleOPoutVsEta->Fill( bestGsfElectron.eta(), bestGsfElectron.eEleClusterOverPout() );
      h2_ele_EeleOPoutVsPhi->Fill( bestGsfElectron.phi(), bestGsfElectron.eEleClusterOverPout() );
      h2_ele_EeleOPoutVsE->Fill( bestGsfElectron.caloEnergy(), bestGsfElectron.eEleClusterOverPout() );
      h1_ele_dEtaSc_propVtx->Fill(bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
      if (bestGsfElectron.ecalDrivenSeed()) h1_ele_dEtaSc_propVtx_eg->Fill(bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
      if (bestGsfElectron.isEB()) h1_ele_dEtaSc_propVtx_barrel->Fill(bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
      if (bestGsfElectron.isEB()&&bestGsfElectron.ecalDrivenSeed()) h1_ele_dEtaSc_propVtx_eg_barrel->Fill(bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
      if (bestGsfElectron.isEE())h1_ele_dEtaSc_propVtx_endcaps->Fill(bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
      if (bestGsfElectron.isEE()&&bestGsfElectron.ecalDrivenSeed()) h1_ele_dEtaSc_propVtx_eg_endcaps->Fill(bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
      h2_ele_dEtaScVsEta_propVtx->Fill( bestGsfElectron.eta(),bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
      h2_ele_dEtaScVsPhi_propVtx->Fill(bestGsfElectron.phi(),bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
      h2_ele_dEtaScVsPt_propVtx->Fill(bestGsfElectron.pt(),bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
      h1_ele_dPhiSc_propVtx->Fill(bestGsfElectron.deltaPhiSuperClusterTrackAtVtx());
      if (bestGsfElectron.ecalDrivenSeed()) h1_ele_dPhiSc_propVtx_eg->Fill(bestGsfElectron.deltaPhiSuperClusterTrackAtVtx());
      if (bestGsfElectron.isEB()) h1_ele_dPhiSc_propVtx_barrel->Fill(bestGsfElectron.deltaPhiSuperClusterTrackAtVtx());
      if (bestGsfElectron.isEB()&&bestGsfElectron.ecalDrivenSeed()) h1_ele_dPhiSc_propVtx_eg_barrel->Fill(bestGsfElectron.deltaPhiSuperClusterTrackAtVtx());
      if (bestGsfElectron.isEE())h1_ele_dPhiSc_propVtx_endcaps->Fill(bestGsfElectron.deltaPhiSuperClusterTrackAtVtx());
      if (bestGsfElectron.isEE()&&bestGsfElectron.ecalDrivenSeed()) h1_ele_dPhiSc_propVtx_eg_endcaps->Fill(bestGsfElectron.deltaPhiSuperClusterTrackAtVtx());
      h2_ele_dPhiScVsEta_propVtx->Fill( bestGsfElectron.eta(),bestGsfElectron.deltaPhiSuperClusterTrackAtVtx());
      h2_ele_dPhiScVsPhi_propVtx->Fill(bestGsfElectron.phi(),bestGsfElectron.deltaPhiSuperClusterTrackAtVtx());
      h2_ele_dPhiScVsPt_propVtx->Fill(bestGsfElectron.pt(),bestGsfElectron.deltaPhiSuperClusterTrackAtVtx());
      h1_ele_dEtaCl_propOut->Fill(bestGsfElectron.deltaEtaSeedClusterTrackAtCalo());
      if (bestGsfElectron.ecalDrivenSeed()) h1_ele_dEtaCl_propOut_eg->Fill(bestGsfElectron.deltaEtaSeedClusterTrackAtCalo());
      if (bestGsfElectron.isEB()) h1_ele_dEtaCl_propOut_barrel->Fill(bestGsfElectron.deltaEtaSeedClusterTrackAtCalo());
      if (bestGsfElectron.isEB()&&bestGsfElectron.ecalDrivenSeed()) h1_ele_dEtaCl_propOut_eg_barrel->Fill(bestGsfElectron.deltaEtaSeedClusterTrackAtCalo());
      if (bestGsfElectron.isEE()) h1_ele_dEtaCl_propOut_endcaps->Fill(bestGsfElectron.deltaEtaSeedClusterTrackAtCalo());
      if (bestGsfElectron.isEE()&&bestGsfElectron.ecalDrivenSeed()) h1_ele_dEtaCl_propOut_eg_endcaps->Fill(bestGsfElectron.deltaEtaSeedClusterTrackAtCalo());
      h2_ele_dEtaClVsEta_propOut->Fill( bestGsfElectron.eta(),bestGsfElectron.deltaEtaSeedClusterTrackAtCalo());
      h2_ele_dEtaClVsPhi_propOut->Fill(bestGsfElectron.phi(),bestGsfElectron.deltaEtaSeedClusterTrackAtCalo());
      h2_ele_dEtaClVsPt_propOut->Fill(bestGsfElectron.pt(),bestGsfElectron.deltaEtaSeedClusterTrackAtCalo());
      h1_ele_dPhiCl_propOut->Fill(bestGsfElectron.deltaPhiSeedClusterTrackAtCalo());
      if (bestGsfElectron.ecalDrivenSeed()) h1_ele_dPhiCl_propOut_eg->Fill(bestGsfElectron.deltaPhiSeedClusterTrackAtCalo());
      if (bestGsfElectron.isEB()) h1_ele_dPhiCl_propOut_barrel->Fill(bestGsfElectron.deltaPhiSeedClusterTrackAtCalo());
      if (bestGsfElectron.isEB()&&bestGsfElectron.ecalDrivenSeed()) h1_ele_dPhiCl_propOut_eg_barrel->Fill(bestGsfElectron.deltaPhiSeedClusterTrackAtCalo());
      if (bestGsfElectron.isEE()) h1_ele_dPhiCl_propOut_endcaps->Fill(bestGsfElectron.deltaPhiSeedClusterTrackAtCalo());
      if (bestGsfElectron.isEE()&&bestGsfElectron.ecalDrivenSeed()) h1_ele_dPhiCl_propOut_eg_endcaps->Fill(bestGsfElectron.deltaPhiSeedClusterTrackAtCalo());
      h2_ele_dPhiClVsEta_propOut->Fill( bestGsfElectron.eta(),bestGsfElectron.deltaPhiSeedClusterTrackAtCalo());
      h2_ele_dPhiClVsPhi_propOut->Fill(bestGsfElectron.phi(),bestGsfElectron.deltaPhiSeedClusterTrackAtCalo());
      h2_ele_dPhiClVsPt_propOut->Fill(bestGsfElectron.pt(),bestGsfElectron.deltaPhiSeedClusterTrackAtCalo());
      h1_ele_dEtaEleCl_propOut->Fill(bestGsfElectron.deltaEtaEleClusterTrackAtCalo());
      if (bestGsfElectron.ecalDrivenSeed()) h1_ele_dEtaEleCl_propOut_eg->Fill(bestGsfElectron.deltaEtaEleClusterTrackAtCalo());
      if (bestGsfElectron.isEB()) h1_ele_dEtaEleCl_propOut_barrel->Fill(bestGsfElectron.deltaEtaEleClusterTrackAtCalo());
      if (bestGsfElectron.isEB()&&bestGsfElectron.ecalDrivenSeed()) h1_ele_dEtaEleCl_propOut_eg_barrel->Fill(bestGsfElectron.deltaEtaEleClusterTrackAtCalo());
      if (bestGsfElectron.isEE()) h1_ele_dEtaEleCl_propOut_endcaps->Fill(bestGsfElectron.deltaEtaEleClusterTrackAtCalo());
      if (bestGsfElectron.isEE()&&bestGsfElectron.ecalDrivenSeed()) h1_ele_dEtaEleCl_propOut_eg_endcaps->Fill(bestGsfElectron.deltaEtaEleClusterTrackAtCalo());
      h2_ele_dEtaEleClVsEta_propOut->Fill( bestGsfElectron.eta(),bestGsfElectron.deltaEtaEleClusterTrackAtCalo());
      h2_ele_dEtaEleClVsPhi_propOut->Fill(bestGsfElectron.phi(),bestGsfElectron.deltaEtaEleClusterTrackAtCalo());
      h2_ele_dEtaEleClVsPt_propOut->Fill(bestGsfElectron.pt(),bestGsfElectron.deltaEtaEleClusterTrackAtCalo());
      h1_ele_dPhiEleCl_propOut->Fill(bestGsfElectron.deltaPhiEleClusterTrackAtCalo());
      if (bestGsfElectron.ecalDrivenSeed()) h1_ele_dPhiEleCl_propOut_eg->Fill(bestGsfElectron.deltaPhiEleClusterTrackAtCalo());
      if (bestGsfElectron.isEB()) h1_ele_dPhiEleCl_propOut_barrel->Fill(bestGsfElectron.deltaPhiEleClusterTrackAtCalo());
      if (bestGsfElectron.isEB()&&bestGsfElectron.ecalDrivenSeed()) h1_ele_dPhiEleCl_propOut_eg_barrel->Fill(bestGsfElectron.deltaPhiEleClusterTrackAtCalo());
      if (bestGsfElectron.isEE()) h1_ele_dPhiEleCl_propOut_endcaps->Fill(bestGsfElectron.deltaPhiEleClusterTrackAtCalo());
      if (bestGsfElectron.isEE()&&bestGsfElectron.ecalDrivenSeed()) h1_ele_dPhiEleCl_propOut_eg_endcaps->Fill(bestGsfElectron.deltaPhiEleClusterTrackAtCalo());
      h2_ele_dPhiEleClVsEta_propOut->Fill( bestGsfElectron.eta(),bestGsfElectron.deltaPhiEleClusterTrackAtCalo());
      h2_ele_dPhiEleClVsPhi_propOut->Fill(bestGsfElectron.phi(),bestGsfElectron.deltaPhiEleClusterTrackAtCalo());
      h2_ele_dPhiEleClVsPt_propOut->Fill(bestGsfElectron.pt(),bestGsfElectron.deltaPhiEleClusterTrackAtCalo());
      h1_ele_HoE->Fill(bestGsfElectron.hadronicOverEm());
      if (bestGsfElectron.ecalDrivenSeed()) h1_ele_HoE_eg->Fill(bestGsfElectron.hadronicOverEm());
      if (bestGsfElectron.isEB()) h1_ele_HoE_barrel->Fill(bestGsfElectron.hadronicOverEm());
      if (bestGsfElectron.isEB()&&bestGsfElectron.ecalDrivenSeed()) h1_ele_HoE_eg_barrel->Fill(bestGsfElectron.hadronicOverEm());
      if (bestGsfElectron.isEE()) h1_ele_HoE_endcaps->Fill(bestGsfElectron.hadronicOverEm());
      if (bestGsfElectron.isEE()&&bestGsfElectron.ecalDrivenSeed()) h1_ele_HoE_eg_endcaps->Fill(bestGsfElectron.hadronicOverEm());
      if (!bestGsfElectron.isEBEtaGap() && !bestGsfElectron.isEBPhiGap()&& !bestGsfElectron.isEBEEGap() &&
          !bestGsfElectron.isEERingGap() && !bestGsfElectron.isEEDeeGap()) h1_ele_HoE_fiducial->Fill(bestGsfElectron.hadronicOverEm());
      h2_ele_HoEVsEta->Fill( bestGsfElectron.eta(),bestGsfElectron.hadronicOverEm());
      h2_ele_HoEVsPhi->Fill(bestGsfElectron.phi(),bestGsfElectron.hadronicOverEm());
      h2_ele_HoEVsE->Fill(bestGsfElectron.caloEnergy(),bestGsfElectron.hadronicOverEm());

      //classes
      int eleClass = bestGsfElectron.classification();
      if (bestGsfElectron.isEE()) eleClass+=10;
      h1_ele_classes->Fill(eleClass);

      h1_ele_eta->Fill(std::abs(bestGsfElectron.eta()));
      if (bestGsfElectron.classification() == GsfElectron::GOLDEN) h1_ele_eta_golden->Fill(std::abs(bestGsfElectron.eta()));
      if (bestGsfElectron.classification() == GsfElectron::BIGBREM) h1_ele_eta_bbrem->Fill(std::abs(bestGsfElectron.eta()));
      if (bestGsfElectron.classification() == GsfElectron::OLDNARROW) h1_ele_eta_narrow->Fill(std::abs(bestGsfElectron.eta()));
      if (bestGsfElectron.classification() == GsfElectron::SHOWERING) h1_ele_eta_shower->Fill(std::abs(bestGsfElectron.eta()));

      //fbrem
      double fbrem_mean=0.;
      if (!readAOD_) // track extra does not exist in AOD
        fbrem_mean =  1. - bestGsfElectron.gsfTrack()->outerMomentum().R()/bestGsfElectron.gsfTrack()->innerMomentum().R();
      double fbrem_mode =  bestGsfElectron.fbrem();
      h1_ele_fbrem->Fill(fbrem_mode);
      p1_ele_fbremVsEta_mode->Fill(bestGsfElectron.eta(),fbrem_mode);
      if (!readAOD_) // track extra does not exist in AOD
        p1_ele_fbremVsEta_mean->Fill(bestGsfElectron.eta(),fbrem_mean);

      if (bestGsfElectron.classification() == GsfElectron::GOLDEN) h2_ele_PinVsPoutGolden_mode->Fill(bestGsfElectron.trackMomentumOut().R(), bestGsfElectron.trackMomentumAtVtx().R());
      if (bestGsfElectron.classification() == GsfElectron::SHOWERING)
        h2_ele_PinVsPoutShowering_mode->Fill(bestGsfElectron.trackMomentumOut().R(), bestGsfElectron.trackMomentumAtVtx().R());
      if (!readAOD_) // track extra does not exist in AOD
        if (bestGsfElectron.classification() == GsfElectron::GOLDEN)
          h2_ele_PinVsPoutGolden_mean->Fill(bestGsfElectron.gsfTrack()->outerMomentum().R(), bestGsfElectron.gsfTrack()->innerMomentum().R());
      if (!readAOD_) // track extra does not exist in AOD
        if (bestGsfElectron.classification() == GsfElectron::SHOWERING)
          h2_ele_PinVsPoutShowering_mean->Fill(bestGsfElectron.gsfTrack()->outerMomentum().R(), bestGsfElectron.gsfTrack()->innerMomentum().R());
      if (bestGsfElectron.classification() == GsfElectron::GOLDEN)
        h2_ele_PtinVsPtoutGolden_mode->Fill(bestGsfElectron.trackMomentumOut().Rho(), bestGsfElectron.trackMomentumAtVtx().Rho());
      if (bestGsfElectron.classification() == GsfElectron::SHOWERING)
        h2_ele_PtinVsPtoutShowering_mode->Fill(bestGsfElectron.trackMomentumOut().Rho(), bestGsfElectron.trackMomentumAtVtx().Rho());
      if (!readAOD_) // track extra does not exist in AOD
        if (bestGsfElectron.classification() == GsfElectron::GOLDEN)
          h2_ele_PtinVsPtoutGolden_mean->Fill(bestGsfElectron.gsfTrack()->outerMomentum().Rho(), bestGsfElectron.gsfTrack()->innerMomentum().Rho());
      if (!readAOD_) // track extra does not exist in AOD
        if (bestGsfElectron.classification() == GsfElectron::SHOWERING)
          h2_ele_PtinVsPtoutShowering_mean->Fill(bestGsfElectron.gsfTrack()->outerMomentum().Rho(), bestGsfElectron.gsfTrack()->innerMomentum().Rho());

      h1_ele_mva->Fill(bestGsfElectron.mva());
      if (bestGsfElectron.ecalDrivenSeed()) h1_ele_provenance->Fill(1.);
      if (bestGsfElectron.trackerDrivenSeed()) h1_ele_provenance->Fill(-1.);
      if (bestGsfElectron.trackerDrivenSeed()||bestGsfElectron.ecalDrivenSeed()) h1_ele_provenance->Fill(0.);
      if (bestGsfElectron.trackerDrivenSeed()&&!bestGsfElectron.ecalDrivenSeed()) h1_ele_provenance->Fill(-2.);
      if (!bestGsfElectron.trackerDrivenSeed()&&bestGsfElectron.ecalDrivenSeed()) h1_ele_provenance->Fill(2.);

      h1_ele_tkSumPt_dr03->Fill(bestGsfElectron.dr03TkSumPt());
      h1_ele_ecalRecHitSumEt_dr03->Fill(bestGsfElectron.dr03EcalRecHitSumEt());
      h1_ele_hcalTowerSumEt_dr03_depth1->Fill(bestGsfElectron.dr03HcalDepth1TowerSumEt());
      h1_ele_hcalTowerSumEt_dr03_depth2->Fill(bestGsfElectron.dr03HcalDepth2TowerSumEt());
      h1_ele_tkSumPt_dr04->Fill(bestGsfElectron.dr04TkSumPt());
      h1_ele_ecalRecHitSumEt_dr04->Fill(bestGsfElectron.dr04EcalRecHitSumEt());
      h1_ele_hcalTowerSumEt_dr04_depth1->Fill(bestGsfElectron.dr04HcalDepth1TowerSumEt());
      h1_ele_hcalTowerSumEt_dr04_depth2->Fill(bestGsfElectron.dr04HcalDepth2TowerSumEt());

     } // gsf electron found

//    } // matching object found

//    }

   } // loop overmatching object

  h1_matchingObjectNum->Fill(matchingObjectNum) ;

 }

void ElectronMcFakeValidator::endJob()
 {
  if (outputFile_!="")
   {
    setStoreFolder("EgammaV/ElectronMcFakeValidator") ;

    // matching object type
    std::string matchingObjectType = "Unknown" ;
    if (std::string::npos!=matchingObjectCollection_.label().find("iterativeCone5GenJets",0))
     { matchingObjectType = "GenJet" ; }

    std::string htitle, xtitle, ytitle ;

    std::cout << "[ElectronMcFakeValidator] efficiency calculation " << std::endl ;
    bookH1andDivide("h_ele_etaEff",h1_ele_matchingObjectEta_matched,h1_matchingObjectEta,"#eta","Efficiency","",true);
    bookH1andDivide("h_ele_zEff",h1_ele_matchingObjectZ_matched,h1_matchingObjectZ,"z (cm)","Efficiency","",true);
    bookH1andDivide("h_ele_absetaEff",h1_ele_matchingObjectAbsEta_matched,h1_matchingObjectAbsEta,"|#eta|","Efficiency");
    bookH1andDivide("h_ele_ptEff",h1_ele_matchingObjectPt_matched,h1_matchingObjectPt,"p_{T} (GeV/c)","Efficiency");
    bookH1andDivide("h_ele_phiEff",h1_ele_matchingObjectPhi_matched,h1_matchingObjectPhi,"#phi (rad)","Efficiency");
//    bookH2andDivide("h_ele_ptEtaEff",h2_ele_matchingObjectPtEta_matched,h2_matchingObjectPtEta,"#eta","p_{T} (GeV/c)");
//
//    std::cout << "[ElectronMcFakeValidator] q-misid calculation " << std::endl;
//    bookH1andDivide("h_ele_etaQmisid",h1_ele_matchingObjectEta_matched_qmisid,h1_simEta,"#eta","q misId","",true);
//    bookH1andDivide("h_ele_zQmisid",h1_ele_matchingObjectZ_matched_qmisid,h1_simZ,"z (cm)","q misId","",true);
//    bookH1andDivide("h_ele_absetaQmisid",h1_ele_matchingObjectAbsEta_matched_qmisid,h1_simAbsEta,"|#eta|","q misId");
//    bookH1andDivide("h_ele_ptQmisid",h1_ele_matchingObjectPt_matched_qmisid,h1_simPt,"p_{T} (GeV/c)","q misId");

    std::cout << "[ElectronMcFakeValidator] all reco electrons " << std::endl ;
    bookH1andDivide("h_ele_etaEff_all",h1_ele_vertexEta_all,h1_matchingObjectEta,"#eta","N_{rec}/N_{gen}","",true);
    bookH1andDivide("h_ele_ptEff_all",h1_ele_vertexPt_all,h1_matchingObjectPt,"p_{T} (GeV/c)","N_{rec}/N_{gen}","",true);

    std::cout << "[ElectronMcFakeValidator] classes " << std::endl ;
    bookH1andDivide("h_ele_eta_goldenFrac",h1_ele_eta_golden,h1_ele_eta,"|#eta|","Fraction of electrons","fraction of golden electrons vs eta");
    bookH1andDivide("h_ele_eta_bbremFrac" ,h1_ele_eta_bbrem ,h1_ele_eta,"|#eta|","Fraction of electrons","fraction of big brem electrons vs eta");
    bookH1andDivide("h_ele_eta_narrowFrac",h1_ele_eta_narrow,h1_ele_eta,"|#eta|","Fraction of electrons","fraction of narrow electrons vs eta");
    bookH1andDivide("h_ele_eta_showerFrac",h1_ele_eta_shower,h1_ele_eta,"|#eta|","Fraction of electrons","fraction of showering electrons vs eta");

    // fbrem
    MonitorElement * h1_ele_xOverX0VsEta = bookH1withSumw2("h_ele_xOverx0VsEta","mean X/X_0 vs eta",eta_nbin/2,0.0,2.5);
    for (int ibin=1;ibin<p1_ele_fbremVsEta_mean->getNbinsX()+1;ibin++) {
      double xOverX0 = 0.;
      if (p1_ele_fbremVsEta_mean->getBinContent(ibin)>0.)
       { xOverX0 = -log(p1_ele_fbremVsEta_mean->getBinContent(ibin)) ; }
      h1_ele_xOverX0VsEta->setBinContent(ibin,xOverX0) ;
    }

    // profiles from 2D histos
    htitle = "Electron / Matching "+matchingObjectType ;
    profileX("h_ele_PoPmatchingObjectVsEta_pfx",h2_ele_PoPmatchingObjectVsEta,htitle+", mean momentum, vs eta","#eta","<P/P_{gen}>");
    profileX("h_ele_PoPmatchingObjectVsPhi_pfx",h2_ele_PoPmatchingObjectVsPhi,htitle+", mean momentum, vs phi","#phi (rad)","<P/P_{gen}>");
    htitle = "Electron - Matching "+matchingObjectType ;
    profileX("h_ele_EtaMnEtamatchingObjectVsEta_pfx",h2_ele_EtaMnEtamatchingObjectVsEta,htitle+", mean eta, vs eta","#eta","<#eta_{rec} - #eta_{gen}>");
    profileX("h_ele_EtaMnEtamatchingObjectVsPhi_pfx",h2_ele_EtaMnEtamatchingObjectVsPhi,htitle+", mean eta, vs phi","#phi (rad)","<#eta_{rec} - #eta_{gen}>");
    profileX("h_ele_PhiMnPhimatchingObjectVsEta_pfx",h2_ele_PhiMnPhimatchingObjectVsEta,htitle+", mean phi, vs eta","#eta","<#phi_{rec} - #phi_{gen}> (rad)");
    profileX("h_ele_PhiMnPhimatchingObjectVsPhi_pfx",h2_ele_PhiMnPhimatchingObjectVsPhi,htitle+", mean phi, vs phi","#phi (rad)","");
    profileX("h_ele_vertexPtVsEta_pfx",h2_ele_vertexPtVsEta,"mean ele transverse momentum vs eta","#eta","<p_{T}> (GeV/c)");
    profileX("h_ele_vertexPtVsPhi_pfx",h2_ele_vertexPtVsPhi,"mean ele transverse momentum vs phi","#phi (rad)","<p_{T}> (GeV/c)");
    profileX("h_ele_EoPVsEta_pfx",h2_ele_EoPVsEta,"mean ele E/p vs eta","#eta","<E/P_{vertex}>");
    profileX("h_ele_EoPVsPhi_pfx",h2_ele_EoPVsPhi,"mean ele E/p vs phi","#phi (rad)","<E/P_{vertex}>");
    profileX("h_ele_EoPoutVsEta_pfx",h2_ele_EoPoutVsEta,"mean ele E/pout vs eta","#eta","<E_{seed}/P_{out}>");
    profileX("h_ele_EoPoutVsPhi_pfx",h2_ele_EoPoutVsPhi,"mean ele E/pout vs phi","#phi (rad)","<E_{seed}/P_{out}>");
    profileX("h_ele_EeleOPoutVsEta_pfx",h2_ele_EeleOPoutVsEta,"mean ele Eele/pout vs eta","#eta","<E_{ele}/P_{out}>");
    profileX("h_ele_EeleOPoutVsPhi_pfx",h2_ele_EeleOPoutVsPhi,"mean ele Eele/pout vs phi","#phi (rad)","<E_{ele}/P_{out}>");
    profileX("h_ele_HoEVsEta_pfx",h2_ele_HoEVsEta,"mean ele H/E vs eta","#eta","<H/E>");
    profileX("h_ele_HoEVsPhi_pfx",h2_ele_HoEVsPhi,"mean ele H/E vs phi","#phi (rad)","<H/E>");
    profileX("h_ele_chi2VsEta_pfx",h2_ele_chi2VsEta,"mean ele track chi2 vs eta","#eta","<#Chi^{2}>");
    profileX("h_ele_chi2VsPhi_pfx",h2_ele_chi2VsPhi,"mean ele track chi2 vs phi","#phi (rad)","<#Chi^{2}>");
    profileX("h_ele_foundHitsVsEta_pfx",h2_ele_foundHitsVsEta,"mean ele track # found hits vs eta","#eta","<N_{hits}>");
    profileX("h_ele_foundHitsVsPhi_pfx",h2_ele_foundHitsVsPhi,"mean ele track # found hits vs phi","#phi (rad)","<N_{hits}>");
    profileX("h_ele_lostHitsVsEta_pfx",h2_ele_lostHitsVsEta,"mean ele track # lost hits vs eta","#eta","<N_{hits}>");
    profileX("h_ele_lostHitsVsPhi_pfx",h2_ele_lostHitsVsPhi,"mean ele track # lost hits vs phi","#phi (rad)","<N_{hits}>");
    profileX("h_ele_vertexTIPVsEta_pfx",h2_ele_vertexTIPVsEta,"mean tip (wrt gen vtx) vs eta","#eta","<TIP> (cm)");
    profileX("h_ele_vertexTIPVsPhi_pfx",h2_ele_vertexTIPVsPhi,"mean tip (wrt gen vtx) vs phi","#phi","<TIP> (cm)");
    profileX("h_ele_vertexTIPVsPt_pfx",h2_ele_vertexTIPVsPt,"mean tip (wrt gen vtx) vs phi","p_{T} (GeV/c)","<TIP> (cm)");

    saveStore(outputFile_) ;
   }
 }


