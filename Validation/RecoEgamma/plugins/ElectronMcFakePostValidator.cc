
#include "Validation/RecoEgamma/plugins/ElectronMcFakePostValidator.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

ElectronMcFakePostValidator::ElectronMcFakePostValidator( const edm::ParameterSet & conf ) 
 : ElectronDqmHarvesterBase(conf) 
 {
  // histos bining and limits

  edm::ParameterSet histosSet = conf.getParameter<edm::ParameterSet>("histosCfg") ;

  set_EfficiencyFlag=histosSet.getParameter<bool>("EfficiencyFlag");
  set_StatOverflowFlag=histosSet.getParameter<bool>("StatOverflowFlag");
 }

ElectronMcFakePostValidator::~ElectronMcFakePostValidator()
 {}

void ElectronMcFakePostValidator::finalize( DQMStore::IBooker & iBooker, DQMStore::IGetter & iGetter )
 {

  setBookIndex(-1) ;
  setBookPrefix("h_ele") ;
  setBookEfficiencyFlag(set_EfficiencyFlag);
  setBookStatOverflowFlag( set_StatOverflowFlag ) ;

  edm::LogInfo("ElectronMcFakePostValidator::finalize") << "efficiency calculation " ;
  bookH1andDivide(iBooker,iGetter, "etaEff","matchingObjectEta_matched","matchingObject_eta","#eta","Efficiency","");
  bookH1andDivide(iBooker,iGetter, "zEff","matchingObjectZ_matched","matchingObject_z","z (cm)","Efficiency","");
  bookH1andDivide(iBooker,iGetter, "absetaEff","matchingObjectAbsEta_matched","matchingObject_abseta","|#eta|","Efficiency","");
  bookH1andDivide(iBooker,iGetter, "ptEff","matchingObjectPt_matched","matchingObject_Pt","p_{T} (GeV/c)","Efficiency","");
  bookH1andDivide(iBooker,iGetter, "phiEff","matchingObjectPhi_matched","matchingObject_phi","#phi (rad)","Efficiency","");/**/
//    bookH2andDivide(iBooker,iGetter, "ptEtaEff","matchingObjectPtEta_matched","matchingObjectPtEta","#eta","p_{T} (GeV/c)","");
//
//    bookH1andDivide(iBooker,iGetter, "etaQmisid","matchingObjectEta_matched_qmisid","h_simEta","#eta","q misId","");
//    bookH1andDivide(iBooker,iGetter, "zQmisid","matchingObjectZ_matched_qmisid","h_simZ","z (cm)","q misId","");
//    bookH1andDivide(iBooker,iGetter, "absetaQmisid","matchingObjectAbsEta_matched_qmisid","h_simAbsEta","|#eta|","q misId","");
//    bookH1andDivide(iBooker,iGetter, "ptQmisid","matchingObjectPt_matched_qmisid","h_simPt","p_{T} (GeV/c)","q misId","");

  edm::LogInfo("ElectronMcFakePostValidator::finalize") << "all reco electrons " ;
  bookH1andDivide(iBooker,iGetter, "etaEff_all","vertexEta_all","matchingObject_eta","#eta","N_{rec}/N_{gen}","");
  bookH1andDivide(iBooker,iGetter, "ptEff_all", "vertexPt_all","matchingObject_Pt","p_{T} (GeV/c)","N_{rec}/N_{gen}","");

  edm::LogInfo("ElectronMcFakePostValidator::finalize") << "classes" ;
  bookH1andDivide(iBooker,iGetter, "eta_goldenFrac","eta_golden","h_ele_eta","|#eta|","Fraction of electrons","fraction of golden electrons vs eta");
  bookH1andDivide(iBooker,iGetter, "eta_bbremFrac" ,"eta_bbrem", "h_ele_eta","|#eta|","Fraction of electrons","fraction of big brem electrons vs eta");
//  bookH1andDivide(iBooker,iGetter, "eta_narrowFrac","eta_narrow","h_ele_eta","|#eta|","Fraction of electrons","fraction of narrow electrons vs eta");
  bookH1andDivide(iBooker,iGetter, "eta_showerFrac","eta_shower","h_ele_eta","|#eta|","Fraction of electrons","fraction of showering electrons vs eta");
/**/
  // fbrem
  MonitorElement * p1_ele_fbremVsEta_mean = get(iGetter, "fbremvsEtamean") ;
  TAxis * etaAxis = p1_ele_fbremVsEta_mean->getTProfile()->GetXaxis() ;
  MonitorElement * h1_ele_xOverX0VsEta = bookH1withSumw2(iBooker, "xOverx0VsEta","mean X/X_0 vs eta",etaAxis->GetNbins(),etaAxis->GetXmin(),etaAxis->GetXmax());
  for (int ibin=1;ibin<etaAxis->GetNbins()+1;ibin++) {
    double xOverX0 = 0.;
    if (p1_ele_fbremVsEta_mean->getBinContent(ibin)>0.)
     { xOverX0 = -log(p1_ele_fbremVsEta_mean->getBinContent(ibin)) ; }
    h1_ele_xOverX0VsEta->setBinContent(ibin,xOverX0) ;
  }
/**/
  // profiles from 2D histos
  profileX(iBooker, iGetter, "PoPmatchingObjectVsEta","","#eta","<P/P_{gen}>");
  profileX(iBooker, iGetter, "PoPmatchingObjectVsPhi","","#phi (rad)","<P/P_{gen}>");
  profileX(iBooker, iGetter, "EtaMnEtamatchingObjectVsEta","","#eta","<#eta_{rec} - #eta_{gen}>");
  profileX(iBooker, iGetter, "EtaMnEtamatchingObjectVsPhi","","#phi (rad)","<#eta_{rec} - #eta_{gen}>");
  profileX(iBooker, iGetter, "PhiMnPhimatchingObjectVsEta","","#eta","<#phi_{rec} - #phi_{gen}> (rad)");
  profileX(iBooker, iGetter, "PhiMnPhimatchingObjectVsPhi","","#phi (rad)","");
  profileX(iBooker, iGetter, "vertexPtVsEta","mean ele transverse momentum vs eta","#eta","<p_{T}> (GeV/c)");
  profileX(iBooker, iGetter, "vertexPtVsPhi","mean ele transverse momentum vs phi","#phi (rad)","<p_{T}> (GeV/c)");
  profileX(iBooker, iGetter, "EoPVsEta","mean ele E/p vs eta","#eta","<E/P_{vertex}>");
  profileX(iBooker, iGetter, "EoPVsPhi","mean ele E/p vs phi","#phi (rad)","<E/P_{vertex}>");
  profileX(iBooker, iGetter, "EoPoutVsEta","mean ele E/pout vs eta","#eta","<E_{seed}/P_{out}>");
  profileX(iBooker, iGetter, "EoPoutVsPhi","mean ele E/pout vs phi","#phi (rad)","<E_{seed}/P_{out}>");
  profileX(iBooker, iGetter, "EeleOPoutVsEta","mean ele Eele/pout vs eta","#eta","<E_{ele}/P_{out}>");
  profileX(iBooker, iGetter, "EeleOPoutVsPhi","mean ele Eele/pout vs phi","#phi (rad)","<E_{ele}/P_{out}>");
  profileX(iBooker, iGetter, "HoEVsEta","mean ele H/E vs eta","#eta","<H/E>");
  profileX(iBooker, iGetter, "HoEVsPhi","mean ele H/E vs phi","#phi (rad)","<H/E>");
  profileX(iBooker, iGetter, "chi2VsEta","mean ele track chi2 vs eta","#eta","<#Chi^{2}>");
  profileX(iBooker, iGetter, "chi2VsPhi","mean ele track chi2 vs phi","#phi (rad)","<#Chi^{2}>");
  profileX(iBooker, iGetter, "foundHitsVsEta","mean ele track # found hits vs eta","#eta","<N_{hits}>");
  profileX(iBooker, iGetter, "foundHitsVsPhi","mean ele track # found hits vs phi","#phi (rad)","<N_{hits}>");
  profileX(iBooker, iGetter, "lostHitsVsEta","mean ele track # lost hits vs eta","#eta","<N_{hits}>");
  profileX(iBooker, iGetter, "lostHitsVsPhi","mean ele track # lost hits vs phi","#phi (rad)","<N_{hits}>");
  profileX(iBooker, iGetter, "vertexTIPVsEta","mean tip (wrt gen vtx) vs eta","#eta","<TIP> (cm)");
  profileX(iBooker, iGetter, "vertexTIPVsPhi","mean tip (wrt gen vtx) vs phi","#phi","<TIP> (cm)");
  profileX(iBooker, iGetter, "vertexTIPVsPt","mean tip (wrt gen vtx) vs phi","p_{T} (GeV/c)","<TIP> (cm)");
  profileX(iBooker, iGetter, "seedDphi2_VsEta","mean ele seed dphi 2nd layer vs eta","#eta","<#phi_{pred} - #phi_{hit}, 2nd layer> (rad)",-0.004,0.004);
  profileX(iBooker, iGetter, "seedDphi2_VsPt","mean ele seed dphi 2nd layer vs pt","p_{T} (GeV/c)","<#phi_{pred} - #phi_{hit}, 2nd layer> (rad)",-0.004,0.004);
  profileX(iBooker, iGetter, "seedDrz2_VsEta","mean ele seed dr(dz) 2nd layer vs eta","#eta","<r(z)_{pred} - r(z)_{hit}, 2nd layer> (cm)",-0.15,0.15);
  profileX(iBooker, iGetter, "seedDrz2_VsPt","mean ele seed dr(dz) 2nd layer vs pt","p_{T} (GeV/c)","<r(z)_{pred} - r(z)_{hit}, 2nd layer> (cm)",-0.15,0.15);
  profileX(iBooker, iGetter, "seedDphi2Pos_VsEta","mean ele seed dphi 2nd layer positron vs eta","#eta","<#phi_{pred} - #phi_{hit}, 2nd layer> (rad)",-0.004,0.004);
  profileX(iBooker, iGetter, "seedDphi2Pos_VsPt","mean ele seed dphi 2nd layer positron vs pt","p_{T} (GeV/c)","<#phi_{pred} - #phi_{hit}, 2nd layer> (rad)",-0.004,0.004);
  profileX(iBooker, iGetter, "seedDrz2Pos_VsEta","mean ele seed dr(dz) 2nd layer positron vs eta","#eta","<r(z)_{pred} - r(z)_{hit}, 2nd layer> (cm)",-0.15,0.15);
  profileX(iBooker, iGetter, "seedDrz2Pos_VsPt","mean ele seed dr(dz) 2nd layer positron vs pt","p_{T} (GeV/c)","<r(z)_{pred} - r(z)_{hit}, 2nd layer> (cm)",-0.15,0.15);
/**/
  }


