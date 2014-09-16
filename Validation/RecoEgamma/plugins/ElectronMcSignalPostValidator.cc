
#include "Validation/RecoEgamma/plugins/ElectronMcSignalPostValidator.h" 
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

ElectronMcSignalPostValidator::ElectronMcSignalPostValidator( const edm::ParameterSet & conf )
 : ElectronDqmHarvesterBase(conf)
 {
  // histos bining and limits

  edm::ParameterSet histosSet = conf.getParameter<edm::ParameterSet>("histosCfg") ;

  set_EfficiencyFlag=histosSet.getParameter<bool>("setEfficiencyFlag");
 }

ElectronMcSignalPostValidator::~ElectronMcSignalPostValidator()
 {}

void ElectronMcSignalPostValidator::book()
 { setBookIndex(-1) ; }/**/

void ElectronMcSignalPostValidator::finalize(DQMStore::IBooker & iBooker)
 {

  setBookPrefix("h_ele") ;

  edm::LogInfo("ElectronMcSignalPostValidator::finalize") << "efficiency calculation" ;
  bookH1andDivide(iBooker, "etaEff","mc_Eta_matched","mc_Eta","#eta","Efficiency","",set_EfficiencyFlag); 
  bookH1andDivide(iBooker, "zEff","mc_Z_matched","mc_Z","z (cm)","Efficiency","",set_EfficiencyFlag);
  bookH1andDivide(iBooker, "absetaEff","mc_AbsEta_matched","mc_AbsEta","|#eta|","Efficiency","",set_EfficiencyFlag);
  bookH1andDivide(iBooker, "ptEff","mc_Pt_matched","mc_Pt","p_{T} (GeV/c)","Efficiency","",set_EfficiencyFlag);
  bookH1andDivide(iBooker, "phiEff","mc_Phi_matched","mc_Phi","#phi (rad)","Efficiency","",set_EfficiencyFlag);
  bookH2andDivide(iBooker, "ptEtaEff","mc_PtEta_matched","mc_PtEta","#eta","p_{T} (GeV/c)","");

  edm::LogInfo("ElectronMcSignalPostValidator::finalize") << "q-misid calculation" ;
  bookH1andDivide(iBooker, "etaQmisid","mc_Eta_matched_qmisid","mc_Eta","#eta","q misId","",set_EfficiencyFlag);
  bookH1andDivide(iBooker, "zQmisid","mc_Z_matched_qmisid","mc_Z","z (cm)","q misId","",set_EfficiencyFlag);
  bookH1andDivide(iBooker, "absetaQmisid","mc_AbsEta_matched_qmisid","mc_AbsEta","|#eta|","q misId","",set_EfficiencyFlag);
  bookH1andDivide(iBooker, "ptQmisid","mc_Pt_matched_qmisid","mc_Pt","p_{T} (GeV/c)","q misId","",set_EfficiencyFlag);

  edm::LogInfo("ElectronMcSignalPostValidator::finalize") << "all reco electrons" ;
  bookH1andDivide(iBooker, "etaEff_all","vertexEta_all","h_mc_Eta","#eta","N_{rec}/N_{gen}","",set_EfficiencyFlag);
  bookH1andDivide(iBooker, "ptEff_all","vertexPt_all","h_mc_Pt","p_{T} (GeV/c)","N_{rec}/N_{gen}","",set_EfficiencyFlag);

  edm::LogInfo("ElectronMcSignalPostValidator::finalize") << "classes" ;
  bookH1andDivide(iBooker, "eta_goldenFrac","eta_golden","h_ele_eta","|#eta|","Fraction of electrons","fraction of golden electrons vs eta",set_EfficiencyFlag);
  bookH1andDivide(iBooker, "eta_bbremFrac" ,"eta_bbrem","h_ele_eta","|#eta|","Fraction of electrons","fraction of big brem electrons vs eta",set_EfficiencyFlag);
  bookH1andDivide(iBooker, "eta_showerFrac","eta_shower","h_ele_eta","|#eta|","Fraction of electrons","fraction of showering electrons vs eta",set_EfficiencyFlag);

  // fbrem
  MonitorElement * p1_ele_fbremVsEta_mean = get("fbremvsEtamean") ;
  TAxis * etaAxis = p1_ele_fbremVsEta_mean->getTProfile()->GetXaxis() ;
  MonitorElement * h1_ele_xOverX0VsEta = bookH1withSumw2(iBooker, "xOverx0VsEta","mean X/X_0 vs eta",etaAxis->GetNbins(),etaAxis->GetXmin(),etaAxis->GetXmax());
  for (int ibin=1;ibin<etaAxis->GetNbins()+1;ibin++) {
    double xOverX0 = 0.;
    if (p1_ele_fbremVsEta_mean->getBinContent(ibin)>0.)
     { xOverX0 = -log(p1_ele_fbremVsEta_mean->getBinContent(ibin)) ; }
    h1_ele_xOverX0VsEta->setBinContent(ibin,xOverX0) ;
  }/**/

  // profiles from 2D histos
  profileX(iBooker, "PoPtrueVsEta","mean ele momentum / gen momentum vs eta","#eta","<P/P_{gen}>");
  profileX(iBooker, "PoPtrueVsPhi","mean ele momentum / gen momentum vs phi","#phi (rad)","<P/P_{gen}>");
  profileX(iBooker, "EoEtruePfVsEg","mean pflow sc energy / true energy vs e/g sc energy","E/E_{gen} (e/g)","<E/E_{gen}> (pflow)") ;
  profileY(iBooker, "EoEtruePfVsEg","mean e/g sc energy / true energy vs pflow sc energy","E/E_{gen} (pflow)","<E/E_{gen}> (eg)") ;
  profileX(iBooker, "EtaMnEtaTrueVsEta","mean ele eta - gen eta vs eta","#eta","<#eta_{rec} - #eta_{gen}>");
  profileX(iBooker, "EtaMnEtaTrueVsPhi","mean ele eta - gen eta vs phi","#phi (rad)","<#eta_{rec} - #eta_{gen}>");
  profileX(iBooker, "PhiMnPhiTrueVsEta","mean ele phi - gen phi vs eta","#eta","<#phi_{rec} - #phi_{gen}> (rad)");
  profileX(iBooker, "PhiMnPhiTrueVsPhi","mean ele phi - gen phi vs phi","#phi (rad)","");
  profileX(iBooker, "vertexPtVsEta","mean ele transverse momentum vs eta","#eta","<p_{T}> (GeV/c)");
  profileX(iBooker, "vertexPtVsPhi","mean ele transverse momentum vs phi","#phi (rad)","<p_{T}> (GeV/c)");
  profileX(iBooker, "EoPVsEta","mean ele E/p vs eta","#eta","<E/P_{vertex}>");
  profileX(iBooker, "EoPVsPhi","mean ele E/p vs phi","#phi (rad)","<E/P_{vertex}>");
  profileX(iBooker, "EoPoutVsEta","mean ele E/pout vs eta","#eta","<E_{seed}/P_{out}>");
  profileX(iBooker, "EoPoutVsPhi","mean ele E/pout vs phi","#phi (rad)","<E_{seed}/P_{out}>");
  profileX(iBooker, "EeleOPoutVsEta","mean ele Eele/pout vs eta","#eta","<E_{ele}/P_{out}>");
  profileX(iBooker, "EeleOPoutVsPhi","mean ele Eele/pout vs phi","#phi (rad)","<E_{ele}/P_{out}>");
  profileX(iBooker, "HoEVsEta","mean ele H/E vs eta","#eta","<H/E>");
  profileX(iBooker, "HoEVsPhi","mean ele H/E vs phi","#phi (rad)","<H/E>");
  profileX(iBooker, "chi2VsEta","mean ele track chi2 vs eta","#eta","<#Chi^{2}>");
  profileX(iBooker, "chi2VsPhi","mean ele track chi2 vs phi","#phi (rad)","<#Chi^{2}>");
  profileX(iBooker, "ambiguousTracksVsEta","mean ele # ambiguous tracks  vs eta","#eta","<N_{ambiguous}>");
  profileX(iBooker, "foundHitsVsEta","mean ele track # found hits vs eta","#eta","<N_{hits}>");
  profileX(iBooker, "foundHitsVsPhi","mean ele track # found hits vs phi","#phi (rad)","<N_{hits}>");
  profileX(iBooker, "lostHitsVsEta","mean ele track # lost hits vs eta","#eta","<N_{hits}>");
  profileX(iBooker, "lostHitsVsPhi","mean ele track # lost hits vs phi","#phi (rad)","<N_{hits}>");
  profileX(iBooker, "vertexTIPVsEta","mean tip (wrt gen vtx) vs eta","#eta","<TIP> (cm)");
  profileX(iBooker, "vertexTIPVsPhi","mean tip (wrt gen vtx) vs phi","#phi","<TIP> (cm)");
  profileX(iBooker, "vertexTIPVsPt","mean tip (wrt gen vtx) vs phi","p_{T} (GeV/c)","<TIP> (cm)");
  profileX(iBooker, "seedDphi2_VsEta","mean ele seed dphi 2nd layer vs eta","#eta","<#phi_{pred} - #phi_{hit}, 2nd layer> (rad)",-0.004,0.004);
  profileX(iBooker, "seedDphi2_VsPt","mean ele seed dphi 2nd layer vs pt","p_{T} (GeV/c)","<#phi_{pred} - #phi_{hit}, 2nd layer> (rad)",-0.004,0.004);
  profileX(iBooker, "seedDrz2_VsEta","mean ele seed dr(dz) 2nd layer vs eta","#eta","<r(z)_{pred} - r(z)_{hit}, 2nd layer> (cm)",-0.15,0.15);
  profileX(iBooker, "seedDrz2_VsPt","mean ele seed dr(dz) 2nd layer vs pt","p_{T} (GeV/c)","<r(z)_{pred} - r(z)_{hit}, 2nd layer> (cm)",-0.15,0.15);
  profileX(iBooker, "seedDphi2Pos_VsEta","mean ele seed dphi 2nd layer positron vs eta","#eta","<#phi_{pred} - #phi_{hit}, 2nd layer> (rad)",-0.004,0.004);
  profileX(iBooker, "seedDphi2Pos_VsPt","mean ele seed dphi 2nd layer positron vs pt","p_{T} (GeV/c)","<#phi_{pred} - #phi_{hit}, 2nd layer> (rad)",-0.004,0.004);
  profileX(iBooker, "seedDrz2Pos_VsEta","mean ele seed dr(dz) 2nd layer positron vs eta","#eta","<r(z)_{pred} - r(z)_{hit}, 2nd layer> (cm)",-0.15,0.15);
  profileX(iBooker, "seedDrz2Pos_VsPt","mean ele seed dr(dz) 2nd layer positron vs pt","p_{T} (GeV/c)","<r(z)_{pred} - r(z)_{hit}, 2nd layer> (cm)",-0.15,0.15);

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


