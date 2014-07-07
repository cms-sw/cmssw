
#include "Validation/RecoEgamma/plugins/ElectronMcFakePostValidator.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

ElectronMcFakePostValidator::ElectronMcFakePostValidator( const edm::ParameterSet & conf )
 : ElectronDqmAnalyzerBase(conf)
 {}

ElectronMcFakePostValidator::~ElectronMcFakePostValidator()
 {}

//void ElectronMcFakePostValidator::book()
void ElectronMcFakePostValidator::bookHistograms( DQMStore::IBooker & iBooker, edm::Run const &, edm::EventSetup const & )
 {
  setBookIndex(-1) ;
  setBookPrefix("h_ele") ;

  edm::LogInfo("ElectronMcFakePostValidator::finalize") << "efficiency calculation " ;
  bookH1andDivide(iBooker, "etaEff","matchingObjectEta_matched","matchingObject_eta","#eta","Efficiency","", "true");
  bookH1andDivide(iBooker, "zEff","matchingObjectZ_matched","matchingObject_z","z (cm)","Efficiency","", "true");
  bookH1andDivide(iBooker, "absetaEff","matchingObjectAbsEta_matched","matchingObject_abseta","|#eta|","Efficiency","", "true");
  bookH1andDivide(iBooker, "ptEff","matchingObjectPt_matched","matchingObject_Pt","p_{T} (GeV/c)","Efficiency","", "true");
  bookH1andDivide(iBooker, "phiEff","matchingObjectPhi_matched","matchingObject_phi","#phi (rad)","Efficiency","", "true");/**/
//    bookH2andDivide(iBooker, "ptEtaEff","matchingObjectPtEta_matched","matchingObjectPtEta","#eta","p_{T} (GeV/c)");
//
//    std::cout << "[ElectronMcFakePostValidator] q-misid calculation " << std::endl;
//    bookH1andDivide(iBooker, "etaQmisid","matchingObjectEta_matched_qmisid","h_simEta","#eta","q misId","",true);
//    bookH1andDivide(iBooker, "zQmisid","matchingObjectZ_matched_qmisid","h_simZ","z (cm)","q misId","",true);
//    bookH1andDivide(iBooker, "absetaQmisid","matchingObjectAbsEta_matched_qmisid","h_simAbsEta","|#eta|","q misId");
//    bookH1andDivide(iBooker, "ptQmisid","matchingObjectPt_matched_qmisid","h_simPt","p_{T} (GeV/c)","q misId");

  edm::LogInfo("ElectronMcFakePostValidator::finalize") << "all reco electrons " ;
  bookH1andDivide(iBooker, "etaEff_all","vertexEta_all","matchingObject_eta","#eta","N_{rec}/N_{gen}");
  bookH1andDivide(iBooker, "ptEff_all", "vertexPt_all","matchingObject_Pt","p_{T} (GeV/c)","N_{rec}/N_{gen}");

  edm::LogInfo("ElectronMcFakePostValidator::finalize") << "classes" ;
  bookH1andDivide(iBooker, "eta_goldenFrac","eta_golden","h_ele_eta","|#eta|","Fraction of electrons","fraction of golden electrons vs eta");
  bookH1andDivide(iBooker, "eta_bbremFrac" ,"eta_bbrem", "h_ele_eta","|#eta|","Fraction of electrons","fraction of big brem electrons vs eta");
//  bookH1andDivide(iBooker, "eta_narrowFrac","eta_narrow","h_ele_eta","|#eta|","Fraction of electrons","fraction of narrow electrons vs eta");
  bookH1andDivide(iBooker, "eta_showerFrac","eta_shower","h_ele_eta","|#eta|","Fraction of electrons","fraction of showering electrons vs eta");

  // fbrem
  MonitorElement * p1_ele_fbremVsEta_mean = ElectronDqmAnalyzerBase::get("fbremvsEtamean") ;
  TAxis * etaAxis = p1_ele_fbremVsEta_mean->getTProfile()->GetXaxis() ;
  h1_ele_xOverX0VsEta = ElectronDqmAnalyzerBase::bookH1withSumw2(iBooker, "xOverx0VsEta","new mean X/X_0 vs eta",etaAxis->GetNbins(),etaAxis->GetXmin(),etaAxis->GetXmax());
  std::cout << "McFakePost bookHistograms h1_ele_xOverX0VsEta : " << h1_ele_xOverX0VsEta->getTH1F()->GetXaxis()->GetNbins()+1 << " Nbins" << std::endl;
  for (int ibin=1;ibin<etaAxis->GetNbins()+1;ibin++) {
    double xOverX0 = 0.; 
	//std::cout << "boucle finalize dans bookHistograms " << ibin << std::endl;
    if (p1_ele_fbremVsEta_mean->getBinContent(ibin)>0.)
     { xOverX0 = -log(p1_ele_fbremVsEta_mean->getBinContent(ibin)) ; }
    h1_ele_xOverX0VsEta->setBinContent(ibin,xOverX0) ;
  }/**/

  // profiles from 2D histos
  ElectronDqmAnalyzerBase::profileX(iBooker,"PoPmatchingObjectVsEta","","#eta","<P/P_{gen}>");
  ElectronDqmAnalyzerBase::profileX(iBooker,"PoPmatchingObjectVsPhi","","#phi (rad)","<P/P_{gen}>");
  ElectronDqmAnalyzerBase::profileX(iBooker,"EtaMnEtamatchingObjectVsEta","","#eta","<#eta_{rec} - #eta_{gen}>");
  ElectronDqmAnalyzerBase::profileX(iBooker,"EtaMnEtamatchingObjectVsPhi","","#phi (rad)","<#eta_{rec} - #eta_{gen}>");
  ElectronDqmAnalyzerBase::profileX(iBooker,"PhiMnPhimatchingObjectVsEta","","#eta","<#phi_{rec} - #phi_{gen}> (rad)");
  ElectronDqmAnalyzerBase::profileX(iBooker,"PhiMnPhimatchingObjectVsPhi","","#phi (rad)","");
  ElectronDqmAnalyzerBase::profileX(iBooker,"vertexPtVsEta","mean ele transverse momentum vs eta","#eta","<p_{T}> (GeV/c)");
  ElectronDqmAnalyzerBase::profileX(iBooker,"vertexPtVsPhi","mean ele transverse momentum vs phi","#phi (rad)","<p_{T}> (GeV/c)");
  ElectronDqmAnalyzerBase::profileX(iBooker,"EoPVsEta","mean ele E/p vs eta","#eta","<E/P_{vertex}>");
  ElectronDqmAnalyzerBase::profileX(iBooker,"EoPVsPhi","mean ele E/p vs phi","#phi (rad)","<E/P_{vertex}>");
  ElectronDqmAnalyzerBase::profileX(iBooker,"EoPoutVsEta","mean ele E/pout vs eta","#eta","<E_{seed}/P_{out}>");
  ElectronDqmAnalyzerBase::profileX(iBooker,"EoPoutVsPhi","mean ele E/pout vs phi","#phi (rad)","<E_{seed}/P_{out}>");
  ElectronDqmAnalyzerBase::profileX(iBooker,"EeleOPoutVsEta","mean ele Eele/pout vs eta","#eta","<E_{ele}/P_{out}>");
  ElectronDqmAnalyzerBase::profileX(iBooker,"EeleOPoutVsPhi","mean ele Eele/pout vs phi","#phi (rad)","<E_{ele}/P_{out}>");
  ElectronDqmAnalyzerBase::profileX(iBooker,"HoEVsEta","mean ele H/E vs eta","#eta","<H/E>");
  ElectronDqmAnalyzerBase::profileX(iBooker,"HoEVsPhi","mean ele H/E vs phi","#phi (rad)","<H/E>");
  ElectronDqmAnalyzerBase::profileX(iBooker,"chi2VsEta","mean ele track chi2 vs eta","#eta","<#Chi^{2}>");
  ElectronDqmAnalyzerBase::profileX(iBooker,"chi2VsPhi","mean ele track chi2 vs phi","#phi (rad)","<#Chi^{2}>");
  ElectronDqmAnalyzerBase::profileX(iBooker,"foundHitsVsEta","mean ele track # found hits vs eta","#eta","<N_{hits}>");
  ElectronDqmAnalyzerBase::profileX(iBooker,"foundHitsVsPhi","mean ele track # found hits vs phi","#phi (rad)","<N_{hits}>");
  ElectronDqmAnalyzerBase::profileX(iBooker,"lostHitsVsEta","mean ele track # lost hits vs eta","#eta","<N_{hits}>");
  ElectronDqmAnalyzerBase::profileX(iBooker,"lostHitsVsPhi","mean ele track # lost hits vs phi","#phi (rad)","<N_{hits}>");
  ElectronDqmAnalyzerBase::profileX(iBooker,"vertexTIPVsEta","mean tip (wrt gen vtx) vs eta","#eta","<TIP> (cm)");
  ElectronDqmAnalyzerBase::profileX(iBooker,"vertexTIPVsPhi","mean tip (wrt gen vtx) vs phi","#phi","<TIP> (cm)");
  ElectronDqmAnalyzerBase::profileX(iBooker,"vertexTIPVsPt","mean tip (wrt gen vtx) vs phi","p_{T} (GeV/c)","<TIP> (cm)");
  ElectronDqmAnalyzerBase::profileX(iBooker,"seedDphi2_VsEta","mean ele seed dphi 2nd layer vs eta","#eta","<#phi_{pred} - #phi_{hit}, 2nd layer> (rad)",-0.004,0.004);
  ElectronDqmAnalyzerBase::profileX(iBooker,"seedDphi2_VsPt","mean ele seed dphi 2nd layer vs pt","p_{T} (GeV/c)","<#phi_{pred} - #phi_{hit}, 2nd layer> (rad)",-0.004,0.004);
  ElectronDqmAnalyzerBase::profileX(iBooker,"seedDrz2_VsEta","mean ele seed dr(dz) 2nd layer vs eta","#eta","<r(z)_{pred} - r(z)_{hit}, 2nd layer> (cm)",-0.15,0.15);
  ElectronDqmAnalyzerBase::profileX(iBooker,"seedDrz2_VsPt","mean ele seed dr(dz) 2nd layer vs pt","p_{T} (GeV/c)","<r(z)_{pred} - r(z)_{hit}, 2nd layer> (cm)",-0.15,0.15);
  ElectronDqmAnalyzerBase::profileX(iBooker,"seedDphi2Pos_VsEta","mean ele seed dphi 2nd layer positron vs eta","#eta","<#phi_{pred} - #phi_{hit}, 2nd layer> (rad)",-0.004,0.004);
  ElectronDqmAnalyzerBase::profileX(iBooker,"seedDphi2Pos_VsPt","mean ele seed dphi 2nd layer positron vs pt","p_{T} (GeV/c)","<#phi_{pred} - #phi_{hit}, 2nd layer> (rad)",-0.004,0.004);
  ElectronDqmAnalyzerBase::profileX(iBooker,"seedDrz2Pos_VsEta","mean ele seed dr(dz) 2nd layer positron vs eta","#eta","<r(z)_{pred} - r(z)_{hit}, 2nd layer> (cm)",-0.15,0.15);
  ElectronDqmAnalyzerBase::profileX(iBooker,"seedDrz2Pos_VsPt","mean ele seed dr(dz) 2nd layer positron vs pt","p_{T} (GeV/c)","<r(z)_{pred} - r(z)_{hit}, 2nd layer> (cm)",-0.15,0.15);
  
 }

void ElectronMcFakePostValidator::finalize()
 {
    // --- AC ---
  std::cout << "McFakePost appel finalize" << std::endl; // to be removed
//  setBookPrefix("h_ele") ;

/*  edm::LogInfo("ElectronMcFakePostValidator::finalize") << "efficiency calculation " ;
  bookH1andDivide(iBooker, "etaEff","matchingObjectEta_matched","matchingObject_eta","#eta","Efficiency");
  bookH1andDivide(iBooker, "zEff","matchingObjectZ_matched","matchingObject_z","z (cm)","Efficiency");
  bookH1andDivide(iBooker, "absetaEff","matchingObjectAbsEta_matched","matchingObject_abseta","|#eta|","Efficiency");
  bookH1andDivide(iBooker, "ptEff","matchingObjectPt_matched","matchingObject_Pt","p_{T} (GeV/c)","Efficiency");
  bookH1andDivide(iBooker, "phiEff","matchingObjectPhi_matched","matchingObject_phi","#phi (rad)","Efficiency");
//    bookH2andDivide(iBooker, "ptEtaEff","matchingObjectPtEta_matched","matchingObjectPtEta","#eta","p_{T} (GeV/c)");
//
//    std::cout << "[ElectronMcFakePostValidator] q-misid calculation " << std::endl;
//    bookH1andDivide(iBooker, "etaQmisid","matchingObjectEta_matched_qmisid","h_simEta","#eta","q misId","",true);
//    bookH1andDivide(iBooker, "zQmisid","matchingObjectZ_matched_qmisid","h_simZ","z (cm)","q misId","",true);
//    bookH1andDivide(iBooker, "absetaQmisid","matchingObjectAbsEta_matched_qmisid","h_simAbsEta","|#eta|","q misId");
//    bookH1andDivide(iBooker, "ptQmisid","matchingObjectPt_matched_qmisid","h_simPt","p_{T} (GeV/c)","q misId");

  edm::LogInfo("ElectronMcFakePostValidator::finalize") << "all reco electrons " ;
  bookH1andDivide(iBooker, "etaEff_all","vertexEta_all","matchingObject_eta","#eta","N_{rec}/N_{gen}");
  bookH1andDivide(iBooker, "ptEff_all", "vertexPt_all","matchingObject_Pt","p_{T} (GeV/c)","N_{rec}/N_{gen}");

  edm::LogInfo("ElectronMcFakePostValidator::finalize") << "classes" ;
  bookH1andDivide(iBooker, "eta_goldenFrac","eta_golden","h_ele_eta","|#eta|","Fraction of electrons","fraction of golden electrons vs eta");
  bookH1andDivide(iBooker, "eta_bbremFrac" ,"eta_bbrem", "h_ele_eta","|#eta|","Fraction of electrons","fraction of big brem electrons vs eta");
//  bookH1andDivide(iBooker, "eta_narrowFrac","eta_narrow","h_ele_eta","|#eta|","Fraction of electrons","fraction of narrow electrons vs eta");
  bookH1andDivide(iBooker, "eta_showerFrac","eta_shower","h_ele_eta","|#eta|","Fraction of electrons","fraction of showering electrons vs eta");*/

  // fbrem
/*  MonitorElement * p1_ele_fbremVsEta_mean = get("fbremvsEtamean") ;
  TAxis * etaAxis = p1_ele_fbremVsEta_mean->getTProfile()->GetXaxis() ;
  MonitorElement * h1_ele_xOverX0VsEta = bookH1withSumw2("xOverx0VsEta","mean X/X_0 vs eta",etaAxis->GetNbins(),etaAxis->GetXmin(),etaAxis->GetXmax());
  for (int ibin=1;ibin<etaAxis->GetNbins()+1;ibin++) {
    double xOverX0 = 0.;
    if (p1_ele_fbremVsEta_mean->getBinContent(ibin)>0.)
     { xOverX0 = -log(p1_ele_fbremVsEta_mean->getBinContent(ibin)) ; }
    h1_ele_xOverX0VsEta->setBinContent(ibin,xOverX0) ;
  }*/

  // profiles from 2D histos
/*  profileX("PoPmatchingObjectVsEta","","#eta","<P/P_{gen}>");
  profileX("PoPmatchingObjectVsPhi","","#phi (rad)","<P/P_{gen}>");
  profileX("EtaMnEtamatchingObjectVsEta","","#eta","<#eta_{rec} - #eta_{gen}>");
  profileX("EtaMnEtamatchingObjectVsPhi","","#phi (rad)","<#eta_{rec} - #eta_{gen}>");
  profileX("PhiMnPhimatchingObjectVsEta","","#eta","<#phi_{rec} - #phi_{gen}> (rad)");
  profileX("PhiMnPhimatchingObjectVsPhi","","#phi (rad)","");
  profileX("vertexPtVsEta","mean ele transverse momentum vs eta","#eta","<p_{T}> (GeV/c)");
  profileX("vertexPtVsPhi","mean ele transverse momentum vs phi","#phi (rad)","<p_{T}> (GeV/c)");
  profileX("EoPVsEta","mean ele E/p vs eta","#eta","<E/P_{vertex}>");
  profileX("EoPVsPhi","mean ele E/p vs phi","#phi (rad)","<E/P_{vertex}>");
  profileX("EoPoutVsEta","mean ele E/pout vs eta","#eta","<E_{seed}/P_{out}>");
  profileX("EoPoutVsPhi","mean ele E/pout vs phi","#phi (rad)","<E_{seed}/P_{out}>");
  profileX("EeleOPoutVsEta","mean ele Eele/pout vs eta","#eta","<E_{ele}/P_{out}>");
  profileX("EeleOPoutVsPhi","mean ele Eele/pout vs phi","#phi (rad)","<E_{ele}/P_{out}>");
  profileX("HoEVsEta","mean ele H/E vs eta","#eta","<H/E>");
  profileX("HoEVsPhi","mean ele H/E vs phi","#phi (rad)","<H/E>");
  profileX("chi2VsEta","mean ele track chi2 vs eta","#eta","<#Chi^{2}>");
  profileX("chi2VsPhi","mean ele track chi2 vs phi","#phi (rad)","<#Chi^{2}>");
  profileX("foundHitsVsEta","mean ele track # found hits vs eta","#eta","<N_{hits}>");
  profileX("foundHitsVsPhi","mean ele track # found hits vs phi","#phi (rad)","<N_{hits}>");
  profileX("lostHitsVsEta","mean ele track # lost hits vs eta","#eta","<N_{hits}>");
  profileX("lostHitsVsPhi","mean ele track # lost hits vs phi","#phi (rad)","<N_{hits}>");
  profileX("vertexTIPVsEta","mean tip (wrt gen vtx) vs eta","#eta","<TIP> (cm)");
  profileX("vertexTIPVsPhi","mean tip (wrt gen vtx) vs phi","#phi","<TIP> (cm)");
  profileX("vertexTIPVsPt","mean tip (wrt gen vtx) vs phi","p_{T} (GeV/c)","<TIP> (cm)");
  profileX("seedDphi2_VsEta","mean ele seed dphi 2nd layer vs eta","#eta","<#phi_{pred} - #phi_{hit}, 2nd layer> (rad)",-0.004,0.004);
  profileX("seedDphi2_VsPt","mean ele seed dphi 2nd layer vs pt","p_{T} (GeV/c)","<#phi_{pred} - #phi_{hit}, 2nd layer> (rad)",-0.004,0.004);
  profileX("seedDrz2_VsEta","mean ele seed dr(dz) 2nd layer vs eta","#eta","<r(z)_{pred} - r(z)_{hit}, 2nd layer> (cm)",-0.15,0.15);
  profileX("seedDrz2_VsPt","mean ele seed dr(dz) 2nd layer vs pt","p_{T} (GeV/c)","<r(z)_{pred} - r(z)_{hit}, 2nd layer> (cm)",-0.15,0.15);
  profileX("seedDphi2Pos_VsEta","mean ele seed dphi 2nd layer positron vs eta","#eta","<#phi_{pred} - #phi_{hit}, 2nd layer> (rad)",-0.004,0.004);
  profileX("seedDphi2Pos_VsPt","mean ele seed dphi 2nd layer positron vs pt","p_{T} (GeV/c)","<#phi_{pred} - #phi_{hit}, 2nd layer> (rad)",-0.004,0.004);
  profileX("seedDrz2Pos_VsEta","mean ele seed dr(dz) 2nd layer positron vs eta","#eta","<r(z)_{pred} - r(z)_{hit}, 2nd layer> (cm)",-0.15,0.15);
  profileX("seedDrz2Pos_VsPt","mean ele seed dr(dz) 2nd layer positron vs pt","p_{T} (GeV/c)","<r(z)_{pred} - r(z)_{hit}, 2nd layer> (cm)",-0.15,0.15);*/
 
  std::cout << "McFakePost appel finalize" << std::endl; // to be removed

}


