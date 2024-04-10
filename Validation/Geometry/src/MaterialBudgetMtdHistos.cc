#include "Validation/Geometry/interface/MaterialBudgetMtdHistos.h"
#include "Validation/Geometry/interface/MaterialBudgetData.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"

MaterialBudgetMtdHistos::MaterialBudgetMtdHistos(std::shared_ptr<MaterialBudgetData> data,
                                                 std::shared_ptr<TestHistoMgr> mgr,
                                                 const std::string& fileName)
    : MaterialBudgetFormat(data), hmgr(mgr) {
  theFileName = fileName;
  book();
}

void MaterialBudgetMtdHistos::book() {
  edm::LogInfo("MaterialBudget") << "MaterialBudgetMtdHistos: Booking user histos";

  static constexpr double minEta = -5.;
  static constexpr double maxEta = 5.;
  static constexpr double minPhi = -3.1416;
  static constexpr double maxPhi = 3.1416;
  static constexpr int nbinEta = 250;
  static constexpr int nbinPhi = 180;

  static constexpr double minEtaBTLZoom = 0.;
  static constexpr double maxEtaBTLZoom = 0.087;
  static constexpr double minPhiBTLZoom = 0.;
  static constexpr double maxPhiBTLZoom = 0.35;
  static constexpr int nbinEtaBTLZoom = 64;
  static constexpr int nbinPhiBTLZoom = 20;

  static constexpr double minMB = 0.;
  static constexpr double maxMBetl = 0.025;
  static constexpr double maxMBbtl = 0.5;
  static constexpr int nbinMB = 25;

  // Material budget: radiation length
  // total X0
  hmgr->addHistoProf1(new TProfile("10", "MB prof Eta [Total];#eta;x/X_{0} ", nbinEta, minEta, maxEta));
  hmgr->addHisto1(new TH1F("11", "Eta ", nbinEta, minEta, maxEta));
  hmgr->addHistoProf1(new TProfile("20", "MB prof Phi [Total];#varphi [rad];x/X_{0} ", nbinPhi, minPhi, maxPhi));
  hmgr->addHisto1(new TH1F("21", "Phi ", nbinPhi, minPhi, maxPhi));
  hmgr->addHistoProf2(new TProfile2D(
      "30", "MB prof Eta  Phi [Total];#eta;#varphi;x/X_{0} ", nbinEta, minEta, maxEta, nbinPhi, minPhi, maxPhi));
  hmgr->addHisto2(new TH2F("31", "Eta vs Phi ", nbinEta, minEta, maxEta, nbinPhi, minPhi, maxPhi));

  // Support
  hmgr->addHistoProf1(new TProfile("110", "MB prof Eta [Support];#eta;x/X_{0}", nbinEta, minEta, maxEta));
  hmgr->addHisto1(new TH1F("111", "Eta [Support]", nbinEta, minEta, maxEta));
  hmgr->addHistoProf1(new TProfile("120", "MB prof Phi [Support];#varphi [rad];x/X_{0}", nbinPhi, minPhi, maxPhi));
  hmgr->addHisto1(new TH1F("121", "Phi [Support]", nbinPhi, minPhi, maxPhi));
  hmgr->addHistoProf2(new TProfile2D(
      "130", "MB prof Eta  Phi [Support];#eta;#varphi;x/X_{0}", nbinEta, minEta, maxEta, nbinPhi, minPhi, maxPhi));
  hmgr->addHisto2(new TH2F("131", "Eta vs Phi [Support]", nbinEta, minEta, maxEta, nbinPhi, minPhi, maxPhi));

  // Sensitive
  hmgr->addHistoProf1(new TProfile("210", "MB prof Eta [Sensitive];#eta;x/X_{0}", nbinEta, minEta, maxEta));
  hmgr->addHisto1(new TH1F("211", "Eta [Sensitive]", nbinEta, minEta, maxEta));
  hmgr->addHistoProf1(new TProfile("220", "MB prof Phi [Sensitive];#varphi [rad];x/X_{0}", nbinPhi, minPhi, maxPhi));
  hmgr->addHisto1(new TH1F("221", "Phi [Sensitive]", nbinPhi, minPhi, maxPhi));
  hmgr->addHistoProf2(new TProfile2D(
      "230", "MB prof Eta  Phi [Sensitive];#eta;#varphi;x/X_{0}", nbinEta, minEta, maxEta, nbinPhi, minPhi, maxPhi));

  hmgr->addHistoProf2(new TProfile2D("10230",
                                     "MB prof Eta  Phi [Sensitive];#eta;#varphi;x/X_{0}",
                                     nbinEtaBTLZoom,
                                     minEtaBTLZoom,
                                     maxEtaBTLZoom,
                                     nbinPhiBTLZoom,
                                     minPhiBTLZoom,
                                     maxPhiBTLZoom));
  hmgr->addHisto2(
      new TH2F("10234", "MB vs Eta [Sensitive];#eta;x/X_{0}", nbinEta, minEta, maxEta, nbinMB, minMB, maxMBetl));
  hmgr->addHisto2(new TH2F(
      "20234", "MB along z vs Eta [Sensitive];#eta;x/X_{0}", nbinEta, minEta, maxEta, nbinMB, minMB, maxMBetl));
  hmgr->addHisto2(
      new TH2F("10235", "MB vs Eta [Sensitive];#eta;x/X_{0}", nbinEta, minEta, maxEta, nbinMB, minMB, maxMBbtl));

  hmgr->addHisto2(new TH2F("231", "Eta vs Phi [Sensitive]", nbinEta, minEta, maxEta, nbinPhi, minPhi, maxPhi));

  // Cables
  hmgr->addHistoProf1(new TProfile("310", "MB prof Eta [Cables];#eta;x/X_{0}", nbinEta, minEta, maxEta));
  hmgr->addHisto1(new TH1F("311", "Eta [Cables]", nbinEta, minEta, maxEta));
  hmgr->addHistoProf1(new TProfile("320", "MB prof Phi [Cables];#varphi [rad];x/X_{0}", nbinPhi, minPhi, maxPhi));
  hmgr->addHisto1(new TH1F("321", "Phi [Cables]", nbinPhi, minPhi, maxPhi));
  hmgr->addHistoProf2(new TProfile2D(
      "330", "MB prof Eta  Phi [Cables];#eta;#varphi;x/X_{0}", nbinEta, minEta, maxEta, nbinPhi, minPhi, maxPhi));
  hmgr->addHisto2(new TH2F("331", "Eta vs Phi [Cables]", nbinEta, minEta, maxEta, nbinPhi, minPhi, maxPhi));

  // Cooling
  hmgr->addHistoProf1(new TProfile("410", "MB prof Eta [Cooling];#eta;x/X_{0}", nbinEta, minEta, maxEta));
  hmgr->addHisto1(new TH1F("411", "Eta [Cooling]", nbinEta, minEta, maxEta));
  hmgr->addHistoProf1(new TProfile("420", "MB prof Phi [Cooling];#varphi [rad];x/X_{0}", nbinPhi, minPhi, maxPhi));
  hmgr->addHisto1(new TH1F("421", "Phi [Cooling]", nbinPhi, minPhi, maxPhi));
  hmgr->addHistoProf2(new TProfile2D(
      "430", "MB prof Eta  Phi [Cooling];#eta;#varphi;x/X_{0}", nbinEta, minEta, maxEta, nbinPhi, minPhi, maxPhi));
  hmgr->addHisto2(new TH2F("431", "Eta vs Phi [Cooling]", nbinEta, minEta, maxEta, nbinPhi, minPhi, maxPhi));

  // Electronics
  hmgr->addHistoProf1(new TProfile("510", "MB prof Eta [Electronics];#eta;x/X_{0}", nbinEta, minEta, maxEta));
  hmgr->addHisto1(new TH1F("511", "Eta [Electronics]", nbinEta, minEta, maxEta));
  hmgr->addHistoProf1(new TProfile("520", "MB prof Phi [Electronics];#varphi [rad];x/X_{0}", nbinPhi, minPhi, maxPhi));
  hmgr->addHisto1(new TH1F("521", "Phi [Electronics]", nbinPhi, minPhi, maxPhi));
  hmgr->addHistoProf2(new TProfile2D(
      "530", "MB prof Eta  Phi [Electronics];#eta;#varphi;x/X_{0}", nbinEta, minEta, maxEta, nbinPhi, minPhi, maxPhi));
  hmgr->addHisto2(new TH2F("531", "Eta vs Phi [Electronics]", nbinEta, minEta, maxEta, nbinPhi, minPhi, maxPhi));

  // Other
  hmgr->addHistoProf1(new TProfile("610", "MB prof Eta [Other];#eta;x/X_{0}", nbinEta, minEta, maxEta));
  hmgr->addHisto1(new TH1F("611", "Eta [Other]", nbinEta, minEta, maxEta));
  hmgr->addHistoProf1(new TProfile("620", "MB prof Phi [Other];#varphi [rad];x/X_{0}", nbinPhi, minPhi, maxPhi));
  hmgr->addHisto1(new TH1F("621", "Phi [Other]", nbinPhi, minPhi, maxPhi));
  hmgr->addHistoProf2(new TProfile2D(
      "630", "MB prof Eta  Phi [Other];#eta;#varphi;x/X_{0}", nbinEta, minEta, maxEta, nbinPhi, minPhi, maxPhi));
  hmgr->addHisto2(new TH2F("631", "Eta vs Phi [Other]", nbinEta, minEta, maxEta, nbinPhi, minPhi, maxPhi));

  // Material budget: interaction length
  // total Lambda0
  hmgr->addHistoProf1(new TProfile("1010", "MB prof Eta [Total];#eta;#lambda/#lambda_{0} ", nbinEta, minEta, maxEta));
  hmgr->addHisto1(new TH1F("1011", "Eta ", nbinEta, minEta, maxEta));
  hmgr->addHistoProf1(
      new TProfile("1020", "MB prof Phi [Total];#varphi [rad];#lambda/#lambda_{0} ", nbinPhi, minPhi, maxPhi));
  hmgr->addHisto1(new TH1F("1021", "Phi ", nbinPhi, minPhi, maxPhi));
  hmgr->addHistoProf2(new TProfile2D("1030",
                                     "MB prof Eta  Phi [Total];#eta;#varphi;#lambda/#lambda_{0} ",
                                     nbinEta,
                                     minEta,
                                     maxEta,
                                     nbinPhi,
                                     minPhi,
                                     maxPhi));
  hmgr->addHisto2(new TH2F("1031", "Eta vs Phi ", nbinEta, minEta, maxEta, nbinPhi, minPhi, maxPhi));

  // Support
  hmgr->addHistoProf1(new TProfile("1110", "MB prof Eta [Support];#eta;#lambda/#lambda_{0}", nbinEta, minEta, maxEta));
  hmgr->addHisto1(new TH1F("1111", "Eta [Support]", nbinEta, minEta, maxEta));
  hmgr->addHistoProf1(
      new TProfile("1120", "MB prof Phi [Support];#varphi [rad];#lambda/#lambda_{0}", nbinPhi, minPhi, maxPhi));
  hmgr->addHisto1(new TH1F("1121", "Phi [Support]", nbinPhi, minPhi, maxPhi));
  hmgr->addHistoProf2(new TProfile2D("1130",
                                     "MB prof Eta  Phi [Support];#eta;#varphi;#lambda/#lambda_{0}",
                                     nbinEta,
                                     minEta,
                                     maxEta,
                                     nbinPhi,
                                     minPhi,
                                     maxPhi));
  hmgr->addHisto2(new TH2F("1131", "Eta vs Phi [Support]", nbinEta, minEta, maxEta, nbinPhi, minPhi, maxPhi));

  // Sensitive
  hmgr->addHistoProf1(
      new TProfile("1210", "MB prof Eta [Sensitive];#eta;#lambda/#lambda_{0}", nbinEta, minEta, maxEta));
  hmgr->addHisto1(new TH1F("1211", "Eta [Sensitive]", nbinEta, minEta, maxEta));
  hmgr->addHistoProf1(
      new TProfile("1220", "MB prof Phi [Sensitive];#varphi [rad];#lambda/#lambda_{0}", nbinPhi, minPhi, maxPhi));
  hmgr->addHisto1(new TH1F("1221", "Phi [Sensitive]", nbinPhi, minPhi, maxPhi));
  hmgr->addHistoProf2(new TProfile2D("1230",
                                     "MB prof Eta  Phi [Sensitive];#eta;#varphi;#lambda/#lambda_{0}",
                                     nbinEta,
                                     minEta,
                                     maxEta,
                                     nbinPhi,
                                     minPhi,
                                     maxPhi));
  hmgr->addHisto2(new TH2F("1231", "Eta vs Phi [Sensitive]", nbinEta, minEta, maxEta, nbinPhi, minPhi, maxPhi));

  // Cables
  hmgr->addHistoProf1(new TProfile("1310", "MB prof Eta [Cables];#eta;#lambda/#lambda_{0}", nbinEta, minEta, maxEta));
  hmgr->addHisto1(new TH1F("1311", "Eta [Cables]", nbinEta, minEta, maxEta));
  hmgr->addHistoProf1(
      new TProfile("1320", "MB prof Phi [Cables];#varphi [rad];#lambda/#lambda_{0}", nbinPhi, minPhi, maxPhi));
  hmgr->addHisto1(new TH1F("1321", "Phi [Cables]", nbinPhi, minPhi, maxPhi));
  hmgr->addHistoProf2(new TProfile2D("1330",
                                     "MB prof Eta  Phi [Cables];#eta;#varphi;#lambda/#lambda_{0}",
                                     nbinEta,
                                     minEta,
                                     maxEta,
                                     nbinPhi,
                                     minPhi,
                                     maxPhi));
  hmgr->addHisto2(new TH2F("1331", "Eta vs Phi [Cables]", nbinEta, minEta, maxEta, nbinPhi, minPhi, maxPhi));

  // Cooling
  hmgr->addHistoProf1(new TProfile("1410", "MB prof Eta [Cooling];#eta;#lambda/#lambda_{0}", nbinEta, minEta, maxEta));
  hmgr->addHisto1(new TH1F("1411", "Eta [Cooling]", nbinEta, minEta, maxEta));
  hmgr->addHistoProf1(
      new TProfile("1420", "MB prof Phi [Cooling];#varphi [rad];#lambda/#lambda_{0}", nbinPhi, minPhi, maxPhi));
  hmgr->addHisto1(new TH1F("1421", "Phi [Cooling]", nbinPhi, minPhi, maxPhi));
  hmgr->addHistoProf2(new TProfile2D("1430",
                                     "MB prof Eta  Phi [Cooling];#eta;#varphi;#lambda/#lambda_{0}",
                                     nbinEta,
                                     minEta,
                                     maxEta,
                                     nbinPhi,
                                     minPhi,
                                     maxPhi));
  hmgr->addHisto2(new TH2F("1431", "Eta vs Phi [Cooling]", nbinEta, minEta, maxEta, nbinPhi, minPhi, maxPhi));

  // Electronics
  hmgr->addHistoProf1(
      new TProfile("1510", "MB prof Eta [Electronics];#eta;#lambda/#lambda_{0}", nbinEta, minEta, maxEta));
  hmgr->addHisto1(new TH1F("1511", "Eta [Electronics]", nbinEta, minEta, maxEta));
  hmgr->addHistoProf1(
      new TProfile("1520", "MB prof Phi [Electronics];#varphi [rad];#lambda/#lambda_{0}", nbinPhi, minPhi, maxPhi));
  hmgr->addHisto1(new TH1F("1521", "Phi [Electronics]", nbinPhi, minPhi, maxPhi));
  hmgr->addHistoProf2(new TProfile2D("1530",
                                     "MB prof Eta  Phi [Electronics];#eta;#varphi;#lambda/#lambda_{0}",
                                     nbinEta,
                                     minEta,
                                     maxEta,
                                     nbinPhi,
                                     minPhi,
                                     maxPhi));
  hmgr->addHisto2(new TH2F("1531", "Eta vs Phi [Electronics]", nbinEta, minEta, maxEta, nbinPhi, minPhi, maxPhi));

  // Other
  hmgr->addHistoProf1(new TProfile("1610", "MB prof Eta [Other];#eta;#lambda/#lambda_{0}", nbinEta, minEta, maxEta));
  hmgr->addHisto1(new TH1F("1611", "Eta [Other]", nbinEta, minEta, maxEta));
  hmgr->addHistoProf1(
      new TProfile("1620", "MB prof Phi [Other];#varphi [rad];#lambda/#lambda_{0}", nbinPhi, minPhi, maxPhi));
  hmgr->addHisto1(new TH1F("1621", "Phi [Other]", nbinPhi, minPhi, maxPhi));
  hmgr->addHistoProf2(new TProfile2D("1630",
                                     "MB prof Eta  Phi [Other];#eta;#varphi;#lambda/#lambda_{0}",
                                     nbinEta,
                                     minEta,
                                     maxEta,
                                     nbinPhi,
                                     minPhi,
                                     maxPhi));
  hmgr->addHisto2(new TH2F("1631", "Eta vs Phi [Other]", nbinEta, minEta, maxEta, nbinPhi, minPhi, maxPhi));

  edm::LogInfo("MaterialBudget") << "MaterialBudgetMtdHistos: booking user histos done";
}

void MaterialBudgetMtdHistos::fillStartTrack() {}

void MaterialBudgetMtdHistos::fillPerStep() {}

void MaterialBudgetMtdHistos::fillEndTrack() {
  // Total X0
  hmgr->getHisto1(11)->Fill(theData->getEta());
  hmgr->getHisto1(21)->Fill(theData->getPhi());
  hmgr->getHisto2(31)->Fill(theData->getEta(), theData->getPhi());

  hmgr->getHistoProf1(10)->Fill(theData->getEta(), theData->getTotalMB());
  hmgr->getHistoProf1(20)->Fill(theData->getPhi(), theData->getTotalMB());
  hmgr->getHistoProf2(30)->Fill(theData->getEta(), theData->getPhi(), theData->getTotalMB());

  // Support
  hmgr->getHisto1(111)->Fill(theData->getEta());
  hmgr->getHisto1(121)->Fill(theData->getPhi());
  hmgr->getHisto2(131)->Fill(theData->getEta(), theData->getPhi());

  hmgr->getHistoProf1(110)->Fill(theData->getEta(), theData->getSupportMB());
  hmgr->getHistoProf1(120)->Fill(theData->getPhi(), theData->getSupportMB());
  hmgr->getHistoProf2(130)->Fill(theData->getEta(), theData->getPhi(), theData->getSupportMB());

  // Sensitive
  hmgr->getHisto1(211)->Fill(theData->getEta());
  hmgr->getHisto1(221)->Fill(theData->getPhi());
  hmgr->getHisto2(231)->Fill(theData->getEta(), theData->getPhi());

  hmgr->getHistoProf1(210)->Fill(theData->getEta(), theData->getSensitiveMB());
  hmgr->getHistoProf1(220)->Fill(theData->getPhi(), theData->getSensitiveMB());
  hmgr->getHistoProf2(230)->Fill(theData->getEta(), theData->getPhi(), theData->getSensitiveMB());

  static constexpr double bfTransitionEta = 1.55;

  if (std::abs(theData->getEta()) > bfTransitionEta) {
    hmgr->getHisto2(10234)->Fill(theData->getEta(), theData->getSensitiveMB());
    double norma = std::tanh(std::abs(theData->getEta()));
    hmgr->getHisto2(20234)->Fill(theData->getEta(), theData->getSensitiveMB() * norma);
  } else {
    hmgr->getHistoProf2(10230)->Fill(theData->getEta(), theData->getPhi(), theData->getSensitiveMB());
    hmgr->getHisto2(10235)->Fill(theData->getEta(), theData->getSensitiveMB());
  }

  // Cables
  hmgr->getHisto1(311)->Fill(theData->getEta());
  hmgr->getHisto1(321)->Fill(theData->getPhi());
  hmgr->getHisto2(331)->Fill(theData->getEta(), theData->getPhi());

  hmgr->getHistoProf1(310)->Fill(theData->getEta(), theData->getCablesMB());
  hmgr->getHistoProf1(320)->Fill(theData->getPhi(), theData->getCablesMB());
  hmgr->getHistoProf2(330)->Fill(theData->getEta(), theData->getPhi(), theData->getCablesMB());

  // Cooling
  hmgr->getHisto1(411)->Fill(theData->getEta());
  hmgr->getHisto1(421)->Fill(theData->getPhi());
  hmgr->getHisto2(431)->Fill(theData->getEta(), theData->getPhi());

  hmgr->getHistoProf1(410)->Fill(theData->getEta(), theData->getCoolingMB());
  hmgr->getHistoProf1(420)->Fill(theData->getPhi(), theData->getCoolingMB());
  hmgr->getHistoProf2(430)->Fill(theData->getEta(), theData->getPhi(), theData->getCoolingMB());

  // Electronics
  hmgr->getHisto1(511)->Fill(theData->getEta());
  hmgr->getHisto1(521)->Fill(theData->getPhi());
  hmgr->getHisto2(531)->Fill(theData->getEta(), theData->getPhi());

  hmgr->getHistoProf1(510)->Fill(theData->getEta(), theData->getElectronicsMB());
  hmgr->getHistoProf1(520)->Fill(theData->getPhi(), theData->getElectronicsMB());
  hmgr->getHistoProf2(530)->Fill(theData->getEta(), theData->getPhi(), theData->getElectronicsMB());

  // Other
  hmgr->getHisto1(611)->Fill(theData->getEta());
  hmgr->getHisto1(621)->Fill(theData->getPhi());
  hmgr->getHisto2(631)->Fill(theData->getEta(), theData->getPhi());

  hmgr->getHistoProf1(610)->Fill(theData->getEta(), theData->getOtherMB());
  hmgr->getHistoProf1(620)->Fill(theData->getPhi(), theData->getOtherMB());
  hmgr->getHistoProf2(630)->Fill(theData->getEta(), theData->getPhi(), theData->getOtherMB());

  // Total Lambda0
  hmgr->getHisto1(1011)->Fill(theData->getEta());
  hmgr->getHisto1(1021)->Fill(theData->getPhi());
  hmgr->getHisto2(1031)->Fill(theData->getEta(), theData->getPhi());

  hmgr->getHistoProf1(1010)->Fill(theData->getEta(), theData->getTotalIL());
  hmgr->getHistoProf1(1020)->Fill(theData->getPhi(), theData->getTotalIL());
  hmgr->getHistoProf2(1030)->Fill(theData->getEta(), theData->getPhi(), theData->getTotalIL());

  // Support
  hmgr->getHisto1(1111)->Fill(theData->getEta());
  hmgr->getHisto1(1121)->Fill(theData->getPhi());
  hmgr->getHisto2(1131)->Fill(theData->getEta(), theData->getPhi());

  hmgr->getHistoProf1(1110)->Fill(theData->getEta(), theData->getSupportIL());
  hmgr->getHistoProf1(1120)->Fill(theData->getPhi(), theData->getSupportIL());
  hmgr->getHistoProf2(1130)->Fill(theData->getEta(), theData->getPhi(), theData->getSupportIL());

  // Sensitive
  hmgr->getHisto1(1211)->Fill(theData->getEta());
  hmgr->getHisto1(1221)->Fill(theData->getPhi());
  hmgr->getHisto2(1231)->Fill(theData->getEta(), theData->getPhi());

  hmgr->getHistoProf1(1210)->Fill(theData->getEta(), theData->getSensitiveIL());
  hmgr->getHistoProf1(1220)->Fill(theData->getPhi(), theData->getSensitiveIL());
  hmgr->getHistoProf2(1230)->Fill(theData->getEta(), theData->getPhi(), theData->getSensitiveIL());

  // Cables
  hmgr->getHisto1(1311)->Fill(theData->getEta());
  hmgr->getHisto1(1321)->Fill(theData->getPhi());
  hmgr->getHisto2(1331)->Fill(theData->getEta(), theData->getPhi());

  hmgr->getHistoProf1(1310)->Fill(theData->getEta(), theData->getCablesIL());
  hmgr->getHistoProf1(1320)->Fill(theData->getPhi(), theData->getCablesIL());
  hmgr->getHistoProf2(1330)->Fill(theData->getEta(), theData->getPhi(), theData->getCablesIL());

  // Cooling
  hmgr->getHisto1(1411)->Fill(theData->getEta());
  hmgr->getHisto1(1421)->Fill(theData->getPhi());
  hmgr->getHisto2(1431)->Fill(theData->getEta(), theData->getPhi());

  hmgr->getHistoProf1(1410)->Fill(theData->getEta(), theData->getCoolingIL());
  hmgr->getHistoProf1(1420)->Fill(theData->getPhi(), theData->getCoolingIL());
  hmgr->getHistoProf2(1430)->Fill(theData->getEta(), theData->getPhi(), theData->getCoolingIL());

  // Electronics
  hmgr->getHisto1(1511)->Fill(theData->getEta());
  hmgr->getHisto1(1521)->Fill(theData->getPhi());
  hmgr->getHisto2(1531)->Fill(theData->getEta(), theData->getPhi());

  hmgr->getHistoProf1(1510)->Fill(theData->getEta(), theData->getElectronicsIL());
  hmgr->getHistoProf1(1520)->Fill(theData->getPhi(), theData->getElectronicsIL());
  hmgr->getHistoProf2(1530)->Fill(theData->getEta(), theData->getPhi(), theData->getElectronicsIL());

  // Other
  hmgr->getHisto1(1611)->Fill(theData->getEta());
  hmgr->getHisto1(1621)->Fill(theData->getPhi());
  hmgr->getHisto2(1631)->Fill(theData->getEta(), theData->getPhi());

  hmgr->getHistoProf1(1610)->Fill(theData->getEta(), theData->getOtherIL());
  hmgr->getHistoProf1(1620)->Fill(theData->getPhi(), theData->getOtherIL());
  hmgr->getHistoProf2(1630)->Fill(theData->getEta(), theData->getPhi(), theData->getOtherIL());
}

void MaterialBudgetMtdHistos::endOfRun() {
  edm::LogInfo("MaterialBudget") << "MaterialBudgetMtdHistos: Writing histos ROOT file to:" << theFileName;
  hmgr->save(theFileName);
}
