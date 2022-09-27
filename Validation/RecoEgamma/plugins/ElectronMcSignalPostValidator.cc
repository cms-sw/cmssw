
#include "Validation/RecoEgamma/plugins/ElectronMcSignalPostValidator.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

ElectronMcSignalPostValidator::ElectronMcSignalPostValidator(const edm::ParameterSet& conf)
    : ElectronDqmHarvesterBase(conf) {
  // histos bining and limits

  edm::ParameterSet histosSet = conf.getParameter<edm::ParameterSet>("histosCfg");

  set_EfficiencyFlag = histosSet.getParameter<bool>("EfficiencyFlag");
  set_StatOverflowFlag = histosSet.getParameter<bool>("StatOverflowFlag");
}

ElectronMcSignalPostValidator::~ElectronMcSignalPostValidator() {}

void ElectronMcSignalPostValidator::finalize(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter) {
  setBookIndex(-1);
  setBookPrefix("h_ele");
  setBookEfficiencyFlag(set_EfficiencyFlag);
  setBookStatOverflowFlag(set_StatOverflowFlag);

  edm::LogInfo("ElectronMcSignalPostValidator::finalize") << "efficiency calculation";
  bookH1andDivide(iBooker,
                  iGetter,
                  "etaEff_Extended",
                  "mc_Eta_Extended_matched",
                  "mc_Eta_Extended",
                  "#eta",
                  "Efficiency",
                  "Efficiency vs gen eta");  //Efficiency vs gen eta --- Eta of matched electrons
  bookH1andDivide(iBooker, iGetter, "zEff", "mc_Z_matched", "mc_Z", "z (cm)", "Efficiency", "");
  bookH1andDivide(iBooker,
                  iGetter,
                  "absetaEff_Extended",
                  "mc_AbsEta_Extended_matched",
                  "mc_AbsEta_Extended",
                  "|#eta|",
                  "Efficiency",
                  "");
  bookH1andDivide(iBooker, iGetter, "ptEff", "mc_Pt_matched", "mc_Pt", "p_{T} (GeV/c)", "Efficiency", "");
  bookH1andDivide(iBooker, iGetter, "phiEff", "mc_Phi_matched", "mc_Phi", "#phi (rad)", "Efficiency", "");

  edm::LogInfo("ElectronMcSignalPostValidator::finalize") << "q-misid calculation";
  bookH1andDivide(iBooker, iGetter, "etaQmisid", "mc_Eta_matched_qmisid", "mc_Eta", "#eta", "q misId", "");
  bookH1andDivide(iBooker, iGetter, "zQmisid", "mc_Z_matched_qmisid", "mc_Z", "z (cm)", "q misId", "");
  bookH1andDivide(iBooker, iGetter, "absetaQmisid", "mc_AbsEta_matched_qmisid", "mc_AbsEta", "|#eta|", "q misId", "");
  bookH1andDivide(iBooker, iGetter, "ptQmisid", "mc_Pt_matched_qmisid", "mc_Pt", "p_{T} (GeV/c)", "q misId", "");

  edm::LogInfo("ElectronMcSignalPostValidator::finalize") << "all reco electrons";
  bookH1andDivide(iBooker, iGetter, "etaEff_all", "vertexEta_all", "h_mc_Eta", "#eta", "N_{rec}/N_{gen}", "");
  bookH1andDivide(iBooker, iGetter, "ptEff_all", "vertexPt_all", "h_mc_Pt", "p_{T} (GeV/c)", "N_{rec}/N_{gen}", "");

  edm::LogInfo("ElectronMcSignalPostValidator::finalize") << "classes";
  bookH1andDivide(iBooker,
                  iGetter,
                  "eta_goldenFrac",
                  "eta_golden",
                  "h_ele_eta",
                  "|#eta|",
                  "Fraction of electrons",
                  "fraction of golden electrons vs eta");
  bookH1andDivide(iBooker,
                  iGetter,
                  "eta_bbremFrac",
                  "eta_bbrem",
                  "h_ele_eta",
                  "|#eta|",
                  "Fraction of electrons",
                  "fraction of big brem electrons vs eta");
  bookH1andDivide(iBooker,
                  iGetter,
                  "eta_showerFrac",
                  "eta_shower",
                  "h_ele_eta",
                  "|#eta|",
                  "Fraction of electrons",
                  "fraction of showering electrons vs eta");

  // fbrem
  MonitorElement* p1_ele_fbremVsEta_mean = get(iGetter, "fbremvsEtamean");
  TAxis* etaAxis = p1_ele_fbremVsEta_mean->getTProfile()->GetXaxis();
  MonitorElement* h1_ele_xOverX0VsEta = bookH1withSumw2(
      iBooker, "xOverx0VsEta", "mean X/X_0 vs eta", etaAxis->GetNbins(), etaAxis->GetXmin(), etaAxis->GetXmax());
  for (int ibin = 1; ibin < etaAxis->GetNbins() + 1; ibin++) {
    double xOverX0 = 0.;
    if (p1_ele_fbremVsEta_mean->getBinContent(ibin) > 0.) {
      xOverX0 = -log(p1_ele_fbremVsEta_mean->getBinContent(ibin));
    }
    h1_ele_xOverX0VsEta->setBinContent(ibin, xOverX0);
  } /**/

  MonitorElement* h1_ele_provenance = get(iGetter, "provenance");
  if (h1_ele_provenance->getBinContent(3) > 0) {
    h1_ele_provenance->getTH1F()->Scale(1. / h1_ele_provenance->getBinContent(3));
  }
  MonitorElement* h1_ele_provenance_barrel = get(iGetter, "provenance_barrel");
  if (h1_ele_provenance_barrel->getBinContent(3) > 0) {
    h1_ele_provenance_barrel->getTH1F()->Scale(1. / h1_ele_provenance_barrel->getBinContent(3));
  }
  MonitorElement* h1_ele_provenance_endcaps = get(iGetter, "provenance_endcaps");
  if (h1_ele_provenance_endcaps->getBinContent(3) > 0) {
    h1_ele_provenance_endcaps->getTH1F()->Scale(1. / h1_ele_provenance_endcaps->getBinContent(3));
  } /**/

  MonitorElement* h1_ele_provenance_Extended = get(iGetter, "provenance_Extended");
  if (h1_ele_provenance_Extended->getBinContent(3) > 0) {
    h1_ele_provenance_Extended->getTH1F()->Scale(1. / h1_ele_provenance_Extended->getBinContent(3));
  }

  // profiles from 2D histos
  profileX(iBooker,
           iGetter,
           "scl_EoEtrueVsrecOfflineVertices",
           "E/Etrue vs number of primary vertices",
           "N_{primary vertices}",
           "E/E_{true}",
           0.8);
  profileX(iBooker,
           iGetter,
           "scl_EoEtrueVsrecOfflineVertices_Extended",
           "E/Etrue vs number of primary vertices, 2.5<|eta|<3",
           "N_{primary vertices}",
           "E/E_{true}",
           0.8);
  profileX(iBooker,
           iGetter,
           "scl_EoEtrueVsrecOfflineVertices_barrel",
           "E/Etrue vs number of primary vertices , barrel",
           "N_{primary vertices}",
           "E/E_{true}",
           0.8);
  profileX(iBooker,
           iGetter,
           "scl_EoEtrueVsrecOfflineVertices_endcaps",
           "E/Etrue vs number of primary vertices , endcaps",
           "N_{primary vertices}",
           "E/E_{true}",
           0.8);

  profileX(iBooker, iGetter, "PoPtrueVsEta_Extended", "mean ele momentum / gen momentum vs eta", "#eta", "<P/P_{gen}>");
  profileX(iBooker, iGetter, "PoPtrueVsPhi", "mean ele momentum / gen momentum vs phi", "#phi (rad)", "<P/P_{gen}>");
  profileX(iBooker, iGetter, "sigmaIetaIetaVsPt", "SigmaIetaIeta vs pt", "p_{T} (GeV/c)", "SigmaIetaIeta");
  profileX(iBooker,
           iGetter,
           "EoEtruePfVsEg",
           "mean mustache SC/true energy vs final SC/true energy",
           "E_{final SC}/E_{gen}",
           "E_{mustache}/E_{gen}");
  profileY(iBooker,
           iGetter,
           "EoEtruePfVsEg",
           "mean mustache SC/true energy vs final SC/true energy",
           "E_{final SC}/E_{gen}",
           "E_{mustache}/E_{gen}");
  profileX(iBooker, iGetter, "EtaMnEtaTrueVsEta", "mean ele eta - gen eta vs eta", "#eta", "<#eta_{rec} - #eta_{gen}>");
  profileX(
      iBooker, iGetter, "EtaMnEtaTrueVsPhi", "mean ele eta - gen eta vs phi", "#phi (rad)", "<#eta_{rec} - #eta_{gen}>");
  profileX(
      iBooker, iGetter, "PhiMnPhiTrueVsEta", "mean ele phi - gen phi vs eta", "#eta", "<#phi_{rec} - #phi_{gen}> (rad)");
  profileX(iBooker, iGetter, "PhiMnPhiTrueVsPhi", "mean ele phi - gen phi vs phi", "#phi (rad)", "");
  profileX(iBooker, iGetter, "vertexPtVsEta", "mean ele transverse momentum vs eta", "#eta", "<p_{T}> (GeV/c)");
  profileX(iBooker, iGetter, "vertexPtVsPhi", "mean ele transverse momentum vs phi", "#phi (rad)", "<p_{T}> (GeV/c)");
  profileX(iBooker, iGetter, "EoPVsEta_Extended", "mean ele E/p vs eta", "#eta", "<E/P_{vertex}>");
  profileX(iBooker, iGetter, "EoPVsPhi", "mean ele E/p vs phi", "#phi (rad)", "<E/P_{vertex}>");
  profileX(iBooker, iGetter, "EoPoutVsEta", "mean ele E/pout vs eta", "#eta", "<E_{seed}/P_{out}>");
  profileX(iBooker, iGetter, "EoPoutVsPhi", "mean ele E/pout vs phi", "#phi (rad)", "<E_{seed}/P_{out}>");
  profileX(iBooker, iGetter, "EeleOPoutVsEta", "mean ele Eele/pout vs eta", "#eta", "<E_{ele}/P_{out}>");
  profileX(iBooker, iGetter, "EeleOPoutVsPhi", "mean ele Eele/pout vs phi", "#phi (rad)", "<E_{ele}/P_{out}>");
  profileX(iBooker, iGetter, "HoEVsEta", "mean ele H/E vs eta", "#eta", "<H/E>");
  profileX(iBooker, iGetter, "HoEVsPhi", "mean ele H/E vs phi", "#phi (rad)", "<H/E>");
  profileX(iBooker, iGetter, "chi2VsEta", "mean ele track chi2 vs eta", "#eta", "<#Chi^{2}>");
  profileX(iBooker, iGetter, "chi2VsPhi", "mean ele track chi2 vs phi", "#phi (rad)", "<#Chi^{2}>");
  profileX(iBooker, iGetter, "ambiguousTracksVsEta", "mean ele # ambiguous tracks  vs eta", "#eta", "<N_{ambiguous}>");
  profileX(iBooker, iGetter, "foundHitsVsEta_Extended", "mean ele track # found hits vs eta", "#eta", "<N_{hits}>");
  profileX(iBooker, iGetter, "foundHitsVsEta_mAOD", "mean ele track # found hits vs eta", "#eta", "<N_{hits}>");
  profileX(iBooker, iGetter, "foundHitsVsPhi", "mean ele track # found hits vs phi", "#phi (rad)", "<N_{hits}>");
  profileX(iBooker, iGetter, "lostHitsVsEta", "mean ele track # lost hits vs eta", "#eta", "<N_{hits}>");
  profileX(iBooker, iGetter, "lostHitsVsPhi", "mean ele track # lost hits vs phi", "#phi (rad)", "<N_{hits}>");
  profileX(iBooker, iGetter, "vertexTIPVsEta", "mean tip (wrt gen vtx) vs eta", "#eta", "<TIP> (cm)");
  profileX(iBooker, iGetter, "vertexTIPVsPhi", "mean tip (wrt gen vtx) vs phi", "#phi", "<TIP> (cm)");
  profileX(iBooker, iGetter, "vertexTIPVsPt", "mean tip (wrt gen vtx) vs phi", "p_{T} (GeV/c)", "<TIP> (cm)");
  profileX(iBooker,
           iGetter,
           "seedDphi2_VsEta",
           "mean ele seed dphi 2nd layer vs eta",
           "#eta",
           "<#phi_{pred} - #phi_{hit}, 2nd layer> (rad)",
           -0.004,
           0.004);
  profileX(iBooker,
           iGetter,
           "seedDphi2_VsPt",
           "mean ele seed dphi 2nd layer vs pt",
           "p_{T} (GeV/c)",
           "<#phi_{pred} - #phi_{hit}, 2nd layer> (rad)",
           -0.004,
           0.004);
  profileX(iBooker,
           iGetter,
           "seedDrz2_VsEta",
           "mean ele seed dr(dz) 2nd layer vs eta",
           "#eta",
           "<r(z)_{pred} - r(z)_{hit}, 2nd layer> (cm)",
           -0.15,
           0.15);
  profileX(iBooker,
           iGetter,
           "seedDrz2_VsPt",
           "mean ele seed dr(dz) 2nd layer vs pt",
           "p_{T} (GeV/c)",
           "<r(z)_{pred} - r(z)_{hit}, 2nd layer> (cm)",
           -0.15,
           0.15);
  profileX(iBooker,
           iGetter,
           "seedDphi2Pos_VsEta",
           "mean ele seed dphi 2nd layer positron vs eta",
           "#eta",
           "<#phi_{pred} - #phi_{hit}, 2nd layer> (rad)",
           -0.004,
           0.004);
  profileX(iBooker,
           iGetter,
           "seedDphi2Pos_VsPt",
           "mean ele seed dphi 2nd layer positron vs pt",
           "p_{T} (GeV/c)",
           "<#phi_{pred} - #phi_{hit}, 2nd layer> (rad)",
           -0.004,
           0.004);
  profileX(iBooker,
           iGetter,
           "seedDrz2Pos_VsEta",
           "mean ele seed dr(dz) 2nd layer positron vs eta",
           "#eta",
           "<r(z)_{pred} - r(z)_{hit}, 2nd layer> (cm)",
           -0.15,
           0.15);
  profileX(iBooker,
           iGetter,
           "seedDrz2Pos_VsPt",
           "mean ele seed dr(dz) 2nd layer positron vs pt",
           "p_{T} (GeV/c)",
           "<r(z)_{pred} - r(z)_{hit}, 2nd layer> (cm)",
           -0.15,
           0.15);
  /**/
}