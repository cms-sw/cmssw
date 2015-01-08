
#include "Validation/RecoEgamma/plugins/ElectronMcSignalPostValidator.h" 
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

ElectronMcSignalPostValidator::ElectronMcSignalPostValidator( const edm::ParameterSet & conf )
 : ElectronDqmHarvesterBase(conf)
 {
  // histos bining and limits

  edm::ParameterSet histosSet = conf.getParameter<edm::ParameterSet>("histosCfg") ;

  set_EfficiencyFlag=histosSet.getParameter<bool>("EfficiencyFlag");
  set_StatOverflowFlag=histosSet.getParameter<bool>("StatOverflowFlag");
 }

ElectronMcSignalPostValidator::~ElectronMcSignalPostValidator()
 {}

void ElectronMcSignalPostValidator::finalize( DQMStore::IBooker & iBooker, DQMStore::IGetter & iGetter )
 {

  setBookIndex(-1) ;
  setBookPrefix("h_ele") ;
  setBookEfficiencyFlag(set_EfficiencyFlag);
  setBookStatOverflowFlag( set_StatOverflowFlag ) ;
  
  edm::LogInfo("ElectronMcSignalPostValidator::finalize") << "efficiency calculation" ;
  bookH1andDivide(iBooker,iGetter, "etaEff","mc_Eta_matched","mc_Eta","#eta","Efficiency",""); 
  bookH1andDivide(iBooker,iGetter, "zEff","mc_Z_matched","mc_Z","z (cm)","Efficiency","");
  bookH1andDivide(iBooker,iGetter, "absetaEff","mc_AbsEta_matched","mc_AbsEta","|#eta|","Efficiency","");
  bookH1andDivide(iBooker,iGetter, "ptEff","mc_Pt_matched","mc_Pt","p_{T} (GeV/c)","Efficiency","");
  bookH1andDivide(iBooker,iGetter, "phiEff","mc_Phi_matched","mc_Phi","#phi (rad)","Efficiency","");
  bookH2andDivide(iBooker,iGetter, "ptEtaEff","mc_PtEta_matched","mc_PtEta","#eta","p_{T} (GeV/c)","");

  edm::LogInfo("ElectronMcSignalPostValidator::finalize") << "q-misid calculation" ;
  bookH1andDivide(iBooker,iGetter, "etaQmisid","mc_Eta_matched_qmisid","mc_Eta","#eta","q misId","");
  bookH1andDivide(iBooker,iGetter, "zQmisid","mc_Z_matched_qmisid","mc_Z","z (cm)","q misId","");
  bookH1andDivide(iBooker,iGetter, "absetaQmisid","mc_AbsEta_matched_qmisid","mc_AbsEta","|#eta|","q misId","");
  bookH1andDivide(iBooker,iGetter, "ptQmisid","mc_Pt_matched_qmisid","mc_Pt","p_{T} (GeV/c)","q misId","");

  edm::LogInfo("ElectronMcSignalPostValidator::finalize") << "all reco electrons" ;
  bookH1andDivide(iBooker,iGetter, "etaEff_all","vertexEta_all","h_mc_Eta","#eta","N_{rec}/N_{gen}","");
  bookH1andDivide(iBooker,iGetter, "ptEff_all","vertexPt_all","h_mc_Pt","p_{T} (GeV/c)","N_{rec}/N_{gen}","");

  edm::LogInfo("ElectronMcSignalPostValidator::finalize") << "classes" ;
  bookH1andDivide(iBooker,iGetter, "eta_goldenFrac","eta_golden","h_ele_eta","|#eta|","Fraction of electrons","fraction of golden electrons vs eta");
  bookH1andDivide(iBooker,iGetter, "eta_bbremFrac" ,"eta_bbrem","h_ele_eta","|#eta|","Fraction of electrons","fraction of big brem electrons vs eta");
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
  }/**/

  // profiles from 2D histos
  profileX(iBooker, iGetter, "PoPtrueVsEta","mean ele momentum / gen momentum vs eta","#eta","<P/P_{gen}>");
  profileX(iBooker, iGetter, "PoPtrueVsPhi","mean ele momentum / gen momentum vs phi","#phi (rad)","<P/P_{gen}>");
  profileX(iBooker, iGetter, "EoEtruePfVsEg","mean pflow sc energy / true energy vs e/g sc energy","E/E_{gen} (e/g)","<E/E_{gen}> (pflow)") ;
  profileY(iBooker, iGetter, "EoEtruePfVsEg","mean e/g sc energy / true energy vs pflow sc energy","E/E_{gen} (pflow)","<E/E_{gen}> (eg)") ;
  profileX(iBooker, iGetter, "EtaMnEtaTrueVsEta","mean ele eta - gen eta vs eta","#eta","<#eta_{rec} - #eta_{gen}>");
  profileX(iBooker, iGetter, "EtaMnEtaTrueVsPhi","mean ele eta - gen eta vs phi","#phi (rad)","<#eta_{rec} - #eta_{gen}>");
  profileX(iBooker, iGetter, "PhiMnPhiTrueVsEta","mean ele phi - gen phi vs eta","#eta","<#phi_{rec} - #phi_{gen}> (rad)");
  profileX(iBooker, iGetter, "PhiMnPhiTrueVsPhi","mean ele phi - gen phi vs phi","#phi (rad)","");
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
  profileX(iBooker, iGetter, "ambiguousTracksVsEta","mean ele # ambiguous tracks  vs eta","#eta","<N_{ambiguous}>");
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

//  // investigation
//  TH2F * h2 = get("PoPtrueVsEta")->getTH2F() ;
//  std::cout<<"H2   entries : "<<h2->GetEntries()<<std::endl ;
//  std::cout<<"H2 effective entries : "<<h2->GetEffectiveEntries()<<std::endl ;
//  Int_t ix, nx = h2->GetNbinsX(), iy, ny = h2->GetNbinsY(), is, nu = 0, no = 0, nb = 0 ;
//  for ( iy = 0 ; iy<=(ny+1) ; ++iy )
//    for ( ix = 0 ; ix<=(nx+1) ; ++ix )
//     {
//      is = iy*(nx+2) + ix ;
//      if (h2->IsBinUnderflow(is)) ++nu ;
//      if (h2->IsBinOverflow(is)) ++no ;
//     }
//  ix = 0 ;
//  for ( iy = 0 ; iy<=(ny+1) ; ++iy )
//   {
//    is = iy*(nx+2) + ix ;
//    nb += (*h2->GetSumw2())[is] ;
//   }
//  ix = nx+1 ;
//  for ( iy = 0 ; iy<=(ny+1) ; ++iy )
//   {
//    is = iy*(nx+2) + ix ;
//    nb += (*h2->GetSumw2())[is] ;
//   }
//  for ( ix = 1 ; ix<=nx ; ++ix )
//   {
//    iy = 0 ;
//    is = iy*(nx+2) + ix ;
//    nb += (*h2->GetSumw2())[is] ;
//    iy = ny+1 ;
//    is = iy*(nx+2) + ix ;
//    nb += (*h2->GetSumw2())[is] ;
//   }
//  std::cout<<"H2   nx      : "<<nx<<std::endl ;
//  std::cout<<"H2   ny      : "<<ny<<std::endl ;
//  std::cout<<"H2   nsumw2  : "<<(*h2->GetSumw2()).fN<<std::endl ;
//  std::cout<<"H2   nu      : "<<nu<<std::endl ;
//  std::cout<<"H2   no      : "<<no<<std::endl ;
//  std::cout<<"H2   outside : "<<nb<<std::endl ;
//  std::cout<<"PFX  entries : "<<h2->ProfileX()->GetEntries()<<std::endl ;
//  std::cout<<"PFX effective entries : "<<h2->ProfileX()->GetEffectiveEntries()<<std::endl ;


}


