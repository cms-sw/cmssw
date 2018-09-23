#include "Validation/Geometry/interface/MaterialBudgetTrackerHistos.h"
#include "Validation/Geometry/interface/MaterialBudgetData.h"

template <class T> const T& max ( const T& a, const T& b ) {
  return (b<a)?a:b;     // or: return comp(b,a)?a:b; for the comp version
}

MaterialBudgetTrackerHistos::MaterialBudgetTrackerHistos(std::shared_ptr<MaterialBudgetData> data,
							 std::shared_ptr<TestHistoMgr> mgr,
							 const std::string& fileName )
  : MaterialBudgetFormat( data ), 
    hmgr(mgr)
{
  theFileName = fileName;
  book();
}


void MaterialBudgetTrackerHistos::book() 
{
  edm::LogInfo("MaterialBudget") << " MaterialBudgetTrackerHistos: Booking Histos";

  // Parameters for 2D histograms
  int nzbin = 1200;
  float zMax = 3000.;
  float zMin = -3000.;
  int nrbin = 290;
  float rMin = -50.;
  float rMax = 1400.;
  
  // total X0
  hmgr->addHistoProf1( new TProfile("10", "MB prof Eta;#eta;x/X_{0} ", 250, -5., 5. ) );
  hmgr->addHisto1( new TH1F("11", "Eta " , 501, -5., 5. ) );
  hmgr->addHistoProf1( new TProfile("20", "MB prof Phi;#varphi [rad];x/X_{0} ", 180, -3.1416, 3.1416 ) );
  hmgr->addHisto1( new TH1F("21", "Phi " , 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf2( new TProfile2D("30", "MB prof Eta  Phi;#eta;#varphi;x/X_{0} ", 250, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHisto2( new TH2F("31", "Eta vs Phi " , 501, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf1( new TProfile("40", "MB prof R;R [mm];x/X_{0} ", 200, 0., 2000. ) );
  hmgr->addHisto1( new TH1F("41", "R " , 200, 0., 2000. ) );
  hmgr->addHistoProf2( new TProfile2D("50", "MB prof sum R  z;z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("999", "Tot track length for MB", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("51", "R vs z " , nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("60", "MB prof local R  z;z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("61", "R vs z " , nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  
  // Support
  hmgr->addHistoProf1( new TProfile("110", "MB prof Eta [Support];#eta;x/X_{0}", 250, -5.0, 5.0 ) );
  hmgr->addHisto1( new TH1F("111", "Eta [Support]" , 501, -5., 5. ) );
  hmgr->addHistoProf1( new TProfile("120", "MB prof Phi [Support];#varphi [rad];x/X_{0}", 180, -3.1416, 3.1416 ) );
  hmgr->addHisto1( new TH1F("121", "Phi [Support]" , 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf2( new TProfile2D("130", "MB prof Eta  Phi [Support];#eta;#varphi;x/X_{0}", 250, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHisto2( new TH2F("131", "Eta vs Phi [Support]" , 501, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf1( new TProfile("140", "MB prof R [Support];R [mm];x/X_{0}", 200, 0., 2000. ) );
  hmgr->addHisto1( new TH1F("141", "R [Support]" , 200, 0., 2000. ) );
  hmgr->addHistoProf2( new TProfile2D("150", "MB prof sum R  z [Support];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("151", "R vs z [Support]" , nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("160", "MB prof local R  z [Support];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("161", "R vs z [Support]" , nzbin, zMin, zMax, nrbin, rMin, rMax ) );

  // Sensitive
  hmgr->addHistoProf1( new TProfile("210", "MB prof Eta [Sensitive];#eta;x/X_{0}", 250, -5.0, 5.0 ) );
  hmgr->addHisto1( new TH1F("211", "Eta [Sensitive]" , 501, -5., 5. ) );
  hmgr->addHistoProf1( new TProfile("220", "MB prof Phi [Sensitive];#varphi [rad];x/X_{0}", 180, -3.1416, 3.1416 ) );
  hmgr->addHisto1( new TH1F("221", "Phi [Sensitive]" , 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf2( new TProfile2D("230", "MB prof Eta  Phi [Sensitive];#eta;#varphi;x/X_{0}", 250, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHisto2( new TH2F("231", "Eta vs Phi [Sensitive]" , 501, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf1( new TProfile("240", "MB prof R [Sensitive];R [mm];x/X_{0}", 200, 0., 2000. ) );
  hmgr->addHisto1( new TH1F("241", "R [Sensitive]" , 200, 0., 2000. ) );
  hmgr->addHistoProf2( new TProfile2D("250", "MB prof sum R  z [Sensitive];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("251", "R vs z [Sensitive]" , nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("260", "MB prof local R  z [Sensitive];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("261", "R vs z [Sensitive]" , nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  // Cables
  hmgr->addHistoProf1( new TProfile("310", "MB prof Eta [Cables];#eta;x/X_{0}", 250, -5.0, 5.0 ) );
  hmgr->addHisto1( new TH1F("311", "Eta [Cables]" , 501, -5., 5. ) );
  hmgr->addHistoProf1( new TProfile("320", "MB prof Phi [Cables];#varphi [rad];x/X_{0}", 180, -3.1416, 3.1416 ) );
  hmgr->addHisto1( new TH1F("321", "Phi [Cables]" , 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf2( new TProfile2D("330", "MB prof Eta  Phi [Cables];#eta;#varphi;x/X_{0}", 250, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHisto2( new TH2F("331", "Eta vs Phi [Cables]" , 501, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf1( new TProfile("340", "MB prof R [Cables];R [mm];x/X_{0}", 200, 0., 2000. ) );
  hmgr->addHisto1( new TH1F("341", "R [Cables]" , 200, 0., 2000. ) );
  hmgr->addHistoProf2( new TProfile2D("350", "MB prof sum R  z [Cables];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("351", "R vs z [Cables]" , nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("360", "MB prof local R  z [Cables];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("361", "R vs z [Cables]" , nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  // Cooling
  hmgr->addHistoProf1( new TProfile("410", "MB prof Eta [Cooling];#eta;x/X_{0}", 250, -5.0, 5.0 ) );
  hmgr->addHisto1( new TH1F("411", "Eta [Cooling]" , 501, -5., 5. ) );
  hmgr->addHistoProf1( new TProfile("420", "MB prof Phi [Cooling];#varphi [rad];x/X_{0}", 180, -3.1416, 3.1416 ) );
  hmgr->addHisto1( new TH1F("421", "Phi [Cooling]" , 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf2( new TProfile2D("430", "MB prof Eta  Phi [Cooling];#eta;#varphi;x/X_{0}", 250, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHisto2( new TH2F("431", "Eta vs Phi [Cooling]" , 501, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf1( new TProfile("440", "MB prof R [Cooling];R [mm];x/X_{0}", 200, 0., 2000. ) );
  hmgr->addHisto1( new TH1F("441", "R [Cooling]" , 200, 0., 2000. ) );
  hmgr->addHistoProf2( new TProfile2D("450", "MB prof sum R  z [Cooling];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("451", "R vs z [Cooling]" , nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("460", "MB prof local R  z [Cooling];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("461", "R vs z [Cooling]" , nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  // Electronics
  hmgr->addHistoProf1( new TProfile("510", "MB prof Eta [Electronics];#eta;x/X_{0}", 250, -5.0, 5.0 ) );
  hmgr->addHisto1( new TH1F("511", "Eta [Electronics]" , 501, -5., 5. ) );
  hmgr->addHistoProf1( new TProfile("520", "MB prof Phi [Electronics];#varphi [rad];x/X_{0}", 180, -3.1416, 3.1416 ) );
  hmgr->addHisto1( new TH1F("521", "Phi [Electronics]" , 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf2( new TProfile2D("530", "MB prof Eta  Phi [Electronics];#eta;#varphi;x/X_{0}", 250, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHisto2( new TH2F("531", "Eta vs Phi [Electronics]" , 501, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf1( new TProfile("540", "MB prof R [Electronics];R [mm];x/X_{0}", 200, 0., 2000. ) );
  hmgr->addHisto1( new TH1F("541", "R [Electronics]" , 200, 0., 2000. ) );
  hmgr->addHistoProf2( new TProfile2D("550", "MB prof sum R  z [Electronics];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("551", "R vs z [Electronics]" , nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("560", "MB prof local R  z [Electronics];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("561", "R vs z [Electronics]" , nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  // Other
  hmgr->addHistoProf1( new TProfile("610", "MB prof Eta [Other];#eta;x/X_{0}", 250, -5.0, 5.0 ) );
  hmgr->addHisto1( new TH1F("611", "Eta [Other]" , 501, -5., 5. ) );
  hmgr->addHistoProf1( new TProfile("620", "MB prof Phi [Other];#varphi [rad];x/X_{0}", 180, -3.1416, 3.1416 ) );
  hmgr->addHisto1( new TH1F("621", "Phi [Other]" , 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf2( new TProfile2D("630", "MB prof Eta  Phi [Other];#eta;#varphi;x/X_{0}", 250, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHisto2( new TH2F("631", "Eta vs Phi [Other]" , 501, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf1( new TProfile("640", "MB prof R [Other];R [mm];x/X_{0}", 200, 0., 2000. ) );
  hmgr->addHisto1( new TH1F("641", "R [Other]" , 200, 0., 2000. ) );
  hmgr->addHistoProf2( new TProfile2D("650", "MB prof sum R  z [Other];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("651", "R vs z [Other]" , nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("660", "MB prof local R  z [Other];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("661", "R vs z [Other]" , nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  // Air
  hmgr->addHistoProf1( new TProfile("710", "MB prof Eta [Air];#eta;x/X_{0}", 250, -5.0, 5.0 ) );
  hmgr->addHisto1( new TH1F("711", "Eta [Air]" , 501, -5., 5. ) );
  hmgr->addHistoProf1( new TProfile("720", "MB prof Phi [Air];#varphi [rad];x/X_{0}", 180, -3.1416, 3.1416 ) );
  hmgr->addHisto1( new TH1F("721", "Phi [Air]" , 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf2( new TProfile2D("730", "MB prof Eta  Phi [Air];#eta;#varphi;x/X_{0}", 250, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHisto2( new TH2F("731", "Eta vs Phi [Air]" , 501, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf1( new TProfile("740", "MB prof R [Air];R [mm];x/X_{0}", 200, 0., 2000. ) );
  hmgr->addHisto1( new TH1F("741", "R [Air]" , 200, 0., 2000. ) );
  hmgr->addHistoProf2( new TProfile2D("750", "MB prof sum R  z [Air];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("751", "R vs z [Air]" , nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("760", "MB prof local R  z [Air];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("761", "R vs z [Air]" , nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  //
  
  // total Lambda0
  hmgr->addHistoProf1( new TProfile("1010", "MB prof Eta;#eta;#lambda/#lambda_{0} ", 250, -5., 5. ) );
  hmgr->addHisto1( new TH1F("1011", "Eta " , 501, -5., 5. ) );
  hmgr->addHistoProf1( new TProfile("1020", "MB prof Phi;#varphi [rad];#lambda/#lambda_{0} ", 180, -3.1416, 3.1416 ) );
  hmgr->addHisto1( new TH1F("1021", "Phi " , 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf2( new TProfile2D("1030", "MB prof Eta  Phi;#eta;#varphi;#lambda/#lambda_{0} ", 250, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHisto2( new TH2F("1031", "Eta vs Phi " , 501, -5., 5., 180, -3.1416, 3.1416 ) );
  
  // rr
  hmgr->addHistoProf1( new TProfile("1040", "MB prof R;R [mm];#lambda/#lambda_{0} ", 200, 0., 2000. ) );
  hmgr->addHisto1( new TH1F("1041", "R " , 200, 0., 2000. ) );
  hmgr->addHistoProf2( new TProfile2D("1050", "MB prof sum R  z;z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("1999", "Tot track length for l0", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("1051", "R vs z " , nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("1060", "MB prof local R  z;z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("1061", "R vs z " , nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  
  // Support
  hmgr->addHistoProf1( new TProfile("1110", "MB prof Eta [Support];#eta;#lambda/#lambda_{0}", 250, -5.0, 5.0 ) );
  hmgr->addHisto1( new TH1F("1111", "Eta [Support]" , 501, -5., 5. ) );
  hmgr->addHistoProf1( new TProfile("1120", "MB prof Phi [Support];#varphi [rad];#lambda/#lambda_{0}", 180, -3.1416, 3.1416 ) );
  hmgr->addHisto1( new TH1F("1121", "Phi [Support]" , 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf2( new TProfile2D("1130", "MB prof Eta  Phi [Support];#eta;#varphi;#lambda/#lambda_{0}", 250, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHisto2( new TH2F("1131", "Eta vs Phi [Support]" , 501, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf1( new TProfile("1140", "MB prof R [Support];R [mm];#lambda/#lambda_{0}", 200, 0., 2000. ) );
  hmgr->addHisto1( new TH1F("1141", "R [Support]" , 200, 0., 2000. ) );
  hmgr->addHistoProf2( new TProfile2D("1150", "MB prof sum R  z [Support];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("1151", "R vs z [Support]" , nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("1160", "MB prof local R  z [Support];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("1161", "R vs z [Support]" , nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  // Sensitive
  hmgr->addHistoProf1( new TProfile("1210", "MB prof Eta [Sensitive];#eta;#lambda/#lambda_{0}", 250, -5.0, 5.0 ) );
  hmgr->addHisto1( new TH1F("1211", "Eta [Sensitive]" , 501, -5., 5. ) );
  hmgr->addHistoProf1( new TProfile("1220", "MB prof Phi [Sensitive];#varphi [rad];#lambda/#lambda_{0}", 180, -3.1416, 3.1416 ) );
  hmgr->addHisto1( new TH1F("1221", "Phi [Sensitive]" , 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf2( new TProfile2D("1230", "MB prof Eta  Phi [Sensitive];#eta;#varphi;#lambda/#lambda_{0}", 250, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHisto2( new TH2F("1231", "Eta vs Phi [Sensitive]" , 501, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf1( new TProfile("1240", "MB prof R [Sensitive];R [mm];#lambda/#lambda_{0}", 200, 0., 2000. ) );
  hmgr->addHisto1( new TH1F("1241", "R [Sensitive]" , 200, 0., 2000. ) );
  hmgr->addHistoProf2( new TProfile2D("1250", "MB prof sum R  z [Sensitive];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("1251", "R vs z [Sensitive]" , nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("1260", "MB prof local R  z [Sensitive];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("1261", "R vs z [Sensitive]" , nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  // Cables
  hmgr->addHistoProf1( new TProfile("1310", "MB prof Eta [Cables];#eta;#lambda/#lambda_{0}", 250, -5.0, 5.0 ) );
  hmgr->addHisto1( new TH1F("1311", "Eta [Cables]" , 501, -5., 5. ) );
  hmgr->addHistoProf1( new TProfile("1320", "MB prof Phi [Cables];#varphi [rad];#lambda/#lambda_{0}", 180, -3.1416, 3.1416 ) );
  hmgr->addHisto1( new TH1F("1321", "Phi [Cables]" , 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf2( new TProfile2D("1330", "MB prof Eta  Phi [Cables];#eta;#varphi;#lambda/#lambda_{0}", 250, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHisto2( new TH2F("1331", "Eta vs Phi [Cables]" , 501, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf1( new TProfile("1340", "MB prof R [Cables];R [mm];#lambda/#lambda_{0}", 200, 0., 2000. ) );
  hmgr->addHisto1( new TH1F("1341", "R [Cables]" , 200, 0., 2000. ) );
  hmgr->addHistoProf2( new TProfile2D("1350", "MB prof sum R  z [Cables];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("1351", "R vs z [Cables]" , nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("1360", "MB prof local R  z [Cables];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("1361", "R vs z [Cables]" , nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  // Cooling
  hmgr->addHistoProf1( new TProfile("1410", "MB prof Eta [Cooling];#eta;#lambda/#lambda_{0}", 250, -5.0, 5.0 ) );
  hmgr->addHisto1( new TH1F("1411", "Eta [Cooling]" , 501, -5., 5. ) );
  hmgr->addHistoProf1( new TProfile("1420", "MB prof Phi [Cooling];#varphi [rad];#lambda/#lambda_{0}", 180, -3.1416, 3.1416 ) );
  hmgr->addHisto1( new TH1F("1421", "Phi [Cooling]" , 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf2( new TProfile2D("1430", "MB prof Eta  Phi [Cooling];#eta;#varphi;#lambda/#lambda_{0}", 250, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHisto2( new TH2F("1431", "Eta vs Phi [Cooling]" , 501, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf1( new TProfile("1440", "MB prof R [Cooling];R [mm];#lambda/#lambda_{0}", 200, 0., 2000. ) );
  hmgr->addHisto1( new TH1F("1441", "R [Cooling]" , 200, 0., 2000. ) );
  hmgr->addHistoProf2( new TProfile2D("1450", "MB prof sum R  z [Cooling];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("1451", "R vs z [Cooling]" , nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("1460", "MB prof local R  z [Cooling];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("1461", "R vs z [Cooling]" , nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  // Electronics
  hmgr->addHistoProf1( new TProfile("1510", "MB prof Eta [Electronics];#eta;#lambda/#lambda_{0}", 250, -5.0, 5.0 ) );
  hmgr->addHisto1( new TH1F("1511", "Eta [Electronics]" , 501, -5., 5. ) );
  hmgr->addHistoProf1( new TProfile("1520", "MB prof Phi [Electronics];#varphi [rad];#lambda/#lambda_{0}", 180, -3.1416, 3.1416 ) );
  hmgr->addHisto1( new TH1F("1521", "Phi [Electronics]" , 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf2( new TProfile2D("1530", "MB prof Eta  Phi [Electronics];#eta;#varphi;#lambda/#lambda_{0}", 250, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHisto2( new TH2F("1531", "Eta vs Phi [Electronics]" , 501, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf1( new TProfile("1540", "MB prof R [Electronics];R [mm];#lambda/#lambda_{0}", 200, 0., 2000. ) );
  hmgr->addHisto1( new TH1F("1541", "R [Electronics]" , 200, 0., 2000. ) );
  hmgr->addHistoProf2( new TProfile2D("1550", "MB prof sum R  z [Electronics];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("1551", "R vs z [Electronics]" , nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("1560", "MB prof local R  z [Electronics];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("1561", "R vs z [Electronics]" , nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  // Other
  hmgr->addHistoProf1( new TProfile("1610", "MB prof Eta [Other];#eta;#lambda/#lambda_{0}", 250, -5.0, 5.0 ) );
  hmgr->addHisto1( new TH1F("1611", "Eta [Other]" , 501, -5., 5. ) );
  hmgr->addHistoProf1( new TProfile("1620", "MB prof Phi [Other];#varphi [rad];#lambda/#lambda_{0}", 180, -3.1416, 3.1416 ) );
  hmgr->addHisto1( new TH1F("1621", "Phi [Other]" , 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf2( new TProfile2D("1630", "MB prof Eta  Phi [Other];#eta;#varphi;#lambda/#lambda_{0}", 250, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHisto2( new TH2F("1631", "Eta vs Phi [Other]" , 501, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf1( new TProfile("1640", "MB prof R [Other];R [mm];#lambda/#lambda_{0}", 200, 0., 2000. ) );
  hmgr->addHisto1( new TH1F("1641", "R [Other]" , 200, 0., 2000. ) );
  hmgr->addHistoProf2( new TProfile2D("1650", "MB prof sum R  z [Other];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("1651", "R vs z [Other]" , nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("1660", "MB prof local R  z [Other];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("1661", "R vs z [Other]" , nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  // Air
  hmgr->addHistoProf1( new TProfile("1710", "MB prof Eta [Air];#eta;#lambda/#lambda_{0}", 250, -5.0, 5.0 ) );
  hmgr->addHisto1( new TH1F("1711", "Eta [Air]" , 501, -5., 5. ) );
  hmgr->addHistoProf1( new TProfile("1720", "MB prof Phi [Air];#varphi [rad];#lambda/#lambda_{0}", 180, -3.1416, 3.1416 ) );
  hmgr->addHisto1( new TH1F("1721", "Phi [Air]" , 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf2( new TProfile2D("1730", "MB prof Eta  Phi [Air];#eta;#varphi;#lambda/#lambda_{0}", 250, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHisto2( new TH2F("1731", "Eta vs Phi [Air]" , 501, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf1( new TProfile("1740", "MB prof R [Air];R [mm];#lambda/#lambda_{0}", 200, 0., 2000. ) );
  hmgr->addHisto1( new TH1F("1741", "R [Air]" , 200, 0., 2000. ) );
  hmgr->addHistoProf2( new TProfile2D("1750", "MB prof sum R  z [Air];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("1751", "R vs z [Air]" , nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("1760", "MB prof local R  z [Air];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("1761", "R vs z [Air]" , nzbin, zMin, zMax, nrbin, rMin, rMax ) );
}


void MaterialBudgetTrackerHistos::fillStartTrack()
{

}


void MaterialBudgetTrackerHistos::fillPerStep()
{

}


void MaterialBudgetTrackerHistos::fillEndTrack()
{
  //
  // fill histograms and profiles only if the material has been crossed
  //
  
  if( theData->getNumberOfSteps() != 0 ) {
    
    // Total X0
    hmgr->getHisto1(11)->Fill(theData->getEta());
    hmgr->getHisto1(21)->Fill(theData->getPhi());
    hmgr->getHisto2(31)->Fill(theData->getEta(),theData->getPhi());
    
    hmgr->getHistoProf1(10)->Fill(theData->getEta(),theData->getTotalMB());
    hmgr->getHistoProf1(20)->Fill(theData->getPhi(),theData->getTotalMB());
    hmgr->getHistoProf2(30)->Fill(theData->getEta(),theData->getPhi(),theData->getTotalMB());
    
    // rr
    
    // Support
    hmgr->getHisto1(111)->Fill(theData->getEta());
    hmgr->getHisto1(121)->Fill(theData->getPhi());
    hmgr->getHisto2(131)->Fill(theData->getEta(),theData->getPhi());
    
    hmgr->getHistoProf1(110)->Fill(theData->getEta(),theData->getSupportMB());
    hmgr->getHistoProf1(120)->Fill(theData->getPhi(),theData->getSupportMB());
    hmgr->getHistoProf2(130)->Fill(theData->getEta(),theData->getPhi(),theData->getSupportMB());
    
    // Sensitive
    hmgr->getHisto1(211)->Fill(theData->getEta());
    hmgr->getHisto1(221)->Fill(theData->getPhi());
    hmgr->getHisto2(231)->Fill(theData->getEta(),theData->getPhi());
    
    hmgr->getHistoProf1(210)->Fill(theData->getEta(),theData->getSensitiveMB());
    hmgr->getHistoProf1(220)->Fill(theData->getPhi(),theData->getSensitiveMB());
    hmgr->getHistoProf2(230)->Fill(theData->getEta(),theData->getPhi(),theData->getSensitiveMB());
    
    // Cables
    hmgr->getHisto1(311)->Fill(theData->getEta());
    hmgr->getHisto1(321)->Fill(theData->getPhi());
    hmgr->getHisto2(331)->Fill(theData->getEta(),theData->getPhi());
    
    hmgr->getHistoProf1(310)->Fill(theData->getEta(),theData->getCablesMB());
    hmgr->getHistoProf1(320)->Fill(theData->getPhi(),theData->getCablesMB());
    hmgr->getHistoProf2(330)->Fill(theData->getEta(),theData->getPhi(),theData->getCablesMB());
    
    // Cooling
    hmgr->getHisto1(411)->Fill(theData->getEta());
    hmgr->getHisto1(421)->Fill(theData->getPhi());
    hmgr->getHisto2(431)->Fill(theData->getEta(),theData->getPhi());
    
    hmgr->getHistoProf1(410)->Fill(theData->getEta(),theData->getCoolingMB());
    hmgr->getHistoProf1(420)->Fill(theData->getPhi(),theData->getCoolingMB());
    hmgr->getHistoProf2(430)->Fill(theData->getEta(),theData->getPhi(),theData->getCoolingMB());
    
    // Electronics
    hmgr->getHisto1(511)->Fill(theData->getEta());
    hmgr->getHisto1(521)->Fill(theData->getPhi());
    hmgr->getHisto2(531)->Fill(theData->getEta(),theData->getPhi());
    
    hmgr->getHistoProf1(510)->Fill(theData->getEta(),theData->getElectronicsMB());
    hmgr->getHistoProf1(520)->Fill(theData->getPhi(),theData->getElectronicsMB());
    hmgr->getHistoProf2(530)->Fill(theData->getEta(),theData->getPhi(),theData->getElectronicsMB());
    
    // Other
    hmgr->getHisto1(611)->Fill(theData->getEta());
    hmgr->getHisto1(621)->Fill(theData->getPhi());
    hmgr->getHisto2(631)->Fill(theData->getEta(),theData->getPhi());
    
    hmgr->getHistoProf1(610)->Fill(theData->getEta(),theData->getOtherMB());
    hmgr->getHistoProf1(620)->Fill(theData->getPhi(),theData->getOtherMB());
    hmgr->getHistoProf2(630)->Fill(theData->getEta(),theData->getPhi(),theData->getOtherMB());
    
    // Air
    hmgr->getHisto1(711)->Fill(theData->getEta());
    hmgr->getHisto1(721)->Fill(theData->getPhi());
    hmgr->getHisto2(731)->Fill(theData->getEta(),theData->getPhi());
    
    hmgr->getHistoProf1(710)->Fill(theData->getEta(),theData->getAirMB());
    hmgr->getHistoProf1(720)->Fill(theData->getPhi(),theData->getAirMB());
    hmgr->getHistoProf2(730)->Fill(theData->getEta(),theData->getPhi(),theData->getAirMB());
    
    //
    // Compute the total x/X0 crossed at each step radius for each path
    //
    //
    float theTotalMB_TOT = 0.0;
    float theTotalMB_SUP = 0.0;
    float theTotalMB_SEN = 0.0;
    float theTotalMB_CAB = 0.0;
    float theTotalMB_COL = 0.0;
    float theTotalMB_ELE = 0.0;
    float theTotalMB_OTH = 0.0;
    float theTotalMB_AIR = 0.0;
    for(int iStep = 0; iStep < theData->getNumberOfSteps(); iStep++) {
      theTotalMB_TOT += theData->getStepDmb(iStep);
      theTotalMB_SUP += theData->getSupportDmb(iStep);
      theTotalMB_SEN += theData->getSensitiveDmb(iStep);
      theTotalMB_CAB += theData->getCablesDmb(iStep);
      theTotalMB_COL += theData->getCoolingDmb(iStep);
      theTotalMB_ELE += theData->getElectronicsDmb(iStep);
      theTotalMB_OTH += theData->getOtherDmb(iStep);
      theTotalMB_AIR += theData->getAirDmb(iStep);        

      int iSup = 0;
      int iSen = 0;
      int iCab = 0;
      int iCol = 0;
      int iEle = 0;
      int iOth = 0;
      int iAir = 0;
      if( theData->getSupportDmb(iStep)>0.     ) { iSup = 1; }
      if( theData->getSensitiveDmb(iStep)>0.   ) { iSen = 1; }
      if( theData->getCablesDmb(iStep)>0.      ) { iCab = 1; }
      if( theData->getCoolingDmb(iStep)>0.     ) { iCol = 1; }
      if( theData->getElectronicsDmb(iStep)>0. ) { iEle = 1; }
      if( theData->getOtherDmb(iStep)>0.       ) { iOth = 1; }
      if( theData->getAirDmb(iStep)>0.         ) { iAir = 1; }

      float deltaRadius = sqrt(
			       pow( theData->getStepFinalX(iStep)-theData->getStepInitialX(iStep),2 )
			       +
			       pow( theData->getStepFinalY(iStep)-theData->getStepInitialY(iStep),2 )
			       );
      float deltaz = theData->getStepFinalZ(iStep)-theData->getStepInitialZ(iStep) ;
      
      float x0 = theData->getStepMaterialX0(iStep);

      int nSubStep = 2;
      float boxWidth = 0.1;
      if( (deltaRadius>boxWidth) || (fabs(deltaz)>boxWidth) ) {
	nSubStep = static_cast<int>(max(
		       ceil(deltaRadius/boxWidth/2.)*2,
		       ceil(fabs(deltaz)/boxWidth/2.)*2
		       ));
      }
      
      for(int iSubStep = 1; iSubStep < nSubStep; iSubStep+=2) {

	float subdeltaRadius = deltaRadius/nSubStep;
	float polarRadius = sqrt(
				 pow( theData->getStepInitialX(iStep),2 )
				 +
				 pow( theData->getStepInitialY(iStep),2 )
				 ) + iSubStep*subdeltaRadius;

	float subdeltaz = deltaz/nSubStep;
	float z = theData->getStepInitialZ(iStep) + iSubStep*subdeltaz;

	float subdelta = sqrt(
			      pow ( subdeltaRadius,2 ) + pow( subdeltaz,2 )
			      );

	float fillValue=subdelta/x0;

	//
	// Average length
	hmgr->getHisto2(999)->Fill(z,polarRadius,subdelta);
	// Total
	hmgr->getHisto1(41)->Fill(polarRadius);
	hmgr->getHistoProf1(40)->Fill(polarRadius,theTotalMB_TOT);
	hmgr->getHisto2(51)->Fill(z,polarRadius);
	hmgr->getHistoProf2(50)->Fill(z,polarRadius,theTotalMB_TOT);
	hmgr->getHisto2(61)->Fill(z,polarRadius);
	hmgr->getHisto2(60)->Fill(z,polarRadius,fillValue);
	// Support
	hmgr->getHisto1(141)->Fill(polarRadius);
	hmgr->getHistoProf1(140)->Fill(polarRadius,theTotalMB_SUP);
	hmgr->getHisto2(151)->Fill(z,polarRadius);
	hmgr->getHistoProf2(150)->Fill(z,polarRadius,theTotalMB_SUP);
	hmgr->getHisto2(161)->Fill(z,polarRadius);
	hmgr->getHisto2(160)->Fill(z,polarRadius,iSup*fillValue);
	// Sensitive
	hmgr->getHisto1(241)->Fill(polarRadius);
	hmgr->getHistoProf1(240)->Fill(polarRadius,theTotalMB_SEN);
	hmgr->getHisto2(251)->Fill(z,polarRadius);
	hmgr->getHistoProf2(250)->Fill(z,polarRadius,theTotalMB_SEN);
	hmgr->getHisto2(261)->Fill(z,polarRadius);
	hmgr->getHisto2(260)->Fill(z,polarRadius,iSen*fillValue);
	// Cables
	hmgr->getHisto1(341)->Fill(polarRadius);
	hmgr->getHistoProf1(340)->Fill(polarRadius,theTotalMB_CAB);
	hmgr->getHisto2(351)->Fill(z,polarRadius);
	hmgr->getHistoProf2(350)->Fill(z,polarRadius,theTotalMB_CAB);
	hmgr->getHisto2(361)->Fill(z,polarRadius);
	hmgr->getHisto2(360)->Fill(z,polarRadius,iCab*fillValue);
	// Cooling
	hmgr->getHisto1(441)->Fill(polarRadius);
	hmgr->getHistoProf1(440)->Fill(polarRadius,theTotalMB_COL);
	hmgr->getHisto2(451)->Fill(z,polarRadius);
	hmgr->getHistoProf2(450)->Fill(z,polarRadius,theTotalMB_COL);
	hmgr->getHisto2(461)->Fill(z,polarRadius);
	hmgr->getHisto2(460)->Fill(z,polarRadius,iCol*fillValue);
	// Electronics
	hmgr->getHisto1(541)->Fill(polarRadius);
	hmgr->getHistoProf1(540)->Fill(polarRadius,theTotalMB_ELE);
	hmgr->getHisto2(551)->Fill(z,polarRadius);
	hmgr->getHistoProf2(550)->Fill(z,polarRadius,theTotalMB_ELE);
	hmgr->getHisto2(561)->Fill(z,polarRadius);
	hmgr->getHisto2(560)->Fill(z,polarRadius,iEle*fillValue);
	// Other
	hmgr->getHisto1(641)->Fill(polarRadius);
	hmgr->getHistoProf1(640)->Fill(polarRadius,theTotalMB_OTH);
	hmgr->getHisto2(651)->Fill(z,polarRadius);
	hmgr->getHistoProf2(650)->Fill(z,polarRadius,theTotalMB_OTH);
	hmgr->getHisto2(661)->Fill(z,polarRadius);
	hmgr->getHisto2(660)->Fill(z,polarRadius,iOth*fillValue);
	// Air
	hmgr->getHisto1(741)->Fill(polarRadius);
	hmgr->getHistoProf1(740)->Fill(polarRadius,theTotalMB_AIR);
	hmgr->getHisto2(751)->Fill(z,polarRadius);
	hmgr->getHistoProf2(750)->Fill(z,polarRadius,theTotalMB_AIR);
	hmgr->getHisto2(761)->Fill(z,polarRadius);
	hmgr->getHisto2(760)->Fill(z,polarRadius,iAir*fillValue);
	//
      }
    }
    //
    //
    ////////////////////////////////////////////////////////////////////////////////////////////
    //
    //
    //

    
    // Total Lambda0
    hmgr->getHisto1(1011)->Fill(theData->getEta());
    hmgr->getHisto1(1021)->Fill(theData->getPhi());
    hmgr->getHisto2(1031)->Fill(theData->getEta(),theData->getPhi());
    
    hmgr->getHistoProf1(1010)->Fill(theData->getEta(),theData->getTotalIL());
    hmgr->getHistoProf1(1020)->Fill(theData->getPhi(),theData->getTotalIL());
    hmgr->getHistoProf2(1030)->Fill(theData->getEta(),theData->getPhi(),theData->getTotalIL());
    
    // Support
    hmgr->getHisto1(1111)->Fill(theData->getEta());
    hmgr->getHisto1(1121)->Fill(theData->getPhi());
    hmgr->getHisto2(1131)->Fill(theData->getEta(),theData->getPhi());
    
    hmgr->getHistoProf1(1110)->Fill(theData->getEta(),theData->getSupportIL());
    hmgr->getHistoProf1(1120)->Fill(theData->getPhi(),theData->getSupportIL());
    hmgr->getHistoProf2(1130)->Fill(theData->getEta(),theData->getPhi(),theData->getSupportIL());
    
    // Sensitive
    hmgr->getHisto1(1211)->Fill(theData->getEta());
    hmgr->getHisto1(1221)->Fill(theData->getPhi());
    hmgr->getHisto2(1231)->Fill(theData->getEta(),theData->getPhi());
    
    hmgr->getHistoProf1(1210)->Fill(theData->getEta(),theData->getSensitiveIL());
    hmgr->getHistoProf1(1220)->Fill(theData->getPhi(),theData->getSensitiveIL());
    hmgr->getHistoProf2(1230)->Fill(theData->getEta(),theData->getPhi(),theData->getSensitiveIL());
    
    // Cables
    hmgr->getHisto1(1311)->Fill(theData->getEta());
    hmgr->getHisto1(1321)->Fill(theData->getPhi());
    hmgr->getHisto2(1331)->Fill(theData->getEta(),theData->getPhi());
    
    hmgr->getHistoProf1(1310)->Fill(theData->getEta(),theData->getCablesIL());
    hmgr->getHistoProf1(1320)->Fill(theData->getPhi(),theData->getCablesIL());
    hmgr->getHistoProf2(1330)->Fill(theData->getEta(),theData->getPhi(),theData->getCablesIL());
    
    // Cooling
    hmgr->getHisto1(1411)->Fill(theData->getEta());
    hmgr->getHisto1(1421)->Fill(theData->getPhi());
    hmgr->getHisto2(1431)->Fill(theData->getEta(),theData->getPhi());
    
    hmgr->getHistoProf1(1410)->Fill(theData->getEta(),theData->getCoolingIL());
    hmgr->getHistoProf1(1420)->Fill(theData->getPhi(),theData->getCoolingIL());
    hmgr->getHistoProf2(1430)->Fill(theData->getEta(),theData->getPhi(),theData->getCoolingIL());
    
    // Electronics
    hmgr->getHisto1(1511)->Fill(theData->getEta());
    hmgr->getHisto1(1521)->Fill(theData->getPhi());
    hmgr->getHisto2(1531)->Fill(theData->getEta(),theData->getPhi());
    
    hmgr->getHistoProf1(1510)->Fill(theData->getEta(),theData->getElectronicsIL());
    hmgr->getHistoProf1(1520)->Fill(theData->getPhi(),theData->getElectronicsIL());
    hmgr->getHistoProf2(1530)->Fill(theData->getEta(),theData->getPhi(),theData->getElectronicsIL());
    
    // Other
    hmgr->getHisto1(1611)->Fill(theData->getEta());
    hmgr->getHisto1(1621)->Fill(theData->getPhi());
    hmgr->getHisto2(1631)->Fill(theData->getEta(),theData->getPhi());
    
    hmgr->getHistoProf1(1610)->Fill(theData->getEta(),theData->getOtherIL());
    hmgr->getHistoProf1(1620)->Fill(theData->getPhi(),theData->getOtherIL());
    hmgr->getHistoProf2(1630)->Fill(theData->getEta(),theData->getPhi(),theData->getOtherIL());
    
    // Air
    hmgr->getHisto1(1711)->Fill(theData->getEta());
    hmgr->getHisto1(1721)->Fill(theData->getPhi());
    hmgr->getHisto2(1731)->Fill(theData->getEta(),theData->getPhi());
    
    hmgr->getHistoProf1(1710)->Fill(theData->getEta(),theData->getAirIL());
    hmgr->getHistoProf1(1720)->Fill(theData->getPhi(),theData->getAirIL());
    hmgr->getHistoProf2(1730)->Fill(theData->getEta(),theData->getPhi(),theData->getAirIL());
    
    // Compute the total l/l0 crossed at each step radius for each path
    float theTotalIL_TOT = 0.0;
    float theTotalIL_SUP = 0.0;
    float theTotalIL_SEN = 0.0;
    float theTotalIL_CAB = 0.0;
    float theTotalIL_COL = 0.0;
    float theTotalIL_ELE = 0.0;
    float theTotalIL_OTH = 0.0;
    float theTotalIL_AIR = 0.0;
    for(int iStep = 0; iStep < theData->getNumberOfSteps(); iStep++) {
      theTotalIL_TOT += theData->getStepDil(iStep);
      theTotalIL_SUP += theData->getSupportDil(iStep);
      theTotalIL_SEN += theData->getSensitiveDil(iStep);
      theTotalIL_CAB += theData->getCablesDil(iStep);
      theTotalIL_COL += theData->getCoolingDil(iStep);
      theTotalIL_ELE += theData->getElectronicsDil(iStep);
      theTotalIL_OTH += theData->getOtherDil(iStep);
      theTotalIL_AIR += theData->getAirDil(iStep);

      int iSup = 0;
      int iSen = 0;
      int iCab = 0;
      int iCol = 0;
      int iEle = 0;
      int iOth = 0;
      int iAir = 0;
      if( theData->getSupportDil(iStep)>0.     ) { iSup = 1; }
      if( theData->getSensitiveDil(iStep)>0.   ) { iSen = 1; }
      if( theData->getCablesDil(iStep)>0.      ) { iCab = 1; }
      if( theData->getCoolingDil(iStep)>0.     ) { iCol = 1; }
      if( theData->getElectronicsDil(iStep)>0. ) { iEle = 1; }
      if( theData->getOtherDil(iStep)>0.       ) { iOth = 1; }
      if( theData->getAirDil(iStep)>0.         ) { iAir = 1; }

      float deltaRadius = sqrt(
			       pow( theData->getStepFinalX(iStep)-theData->getStepInitialX(iStep),2 )
			       +
			       pow( theData->getStepFinalY(iStep)-theData->getStepInitialY(iStep),2 )
			       );
      float deltaz = theData->getStepFinalZ(iStep)-theData->getStepInitialZ(iStep) ;
      
      float il = theData->getStepMaterialLambda0(iStep);

      int nSubStep = 2;
      float boxWidth = 0.1;
      if( (deltaRadius>boxWidth) || (fabs(deltaz)>boxWidth) ) {
	nSubStep = static_cast<int>(max(
		       ceil(deltaRadius/boxWidth/2.)*2,
		       ceil(fabs(deltaz)/boxWidth/2.)*2
		       ));
      }
      
      for(int iSubStep = 1; iSubStep < nSubStep; iSubStep+=2) {

	float subdeltaRadius = deltaRadius/nSubStep;
	float polarRadius = sqrt(
				 pow( theData->getStepInitialX(iStep),2 )
				 +
				 pow( theData->getStepInitialY(iStep),2 )
				 ) + iSubStep*subdeltaRadius;

	float subdeltaz = deltaz/nSubStep;
	float z = theData->getStepInitialZ(iStep) + iSubStep*subdeltaz;

	float subdelta = sqrt(
			      pow ( subdeltaRadius,2 ) + pow( subdeltaz,2 )
			      );

	float fillValue=subdelta/il;

	//
	// Average length
	hmgr->getHisto2(1999)->Fill(z,polarRadius,subdelta);
	// Total
	hmgr->getHisto1(1041)->Fill(polarRadius);
	hmgr->getHistoProf1(1040)->Fill(polarRadius,theTotalIL_TOT);
	hmgr->getHisto2(1051)->Fill(z,polarRadius);
	hmgr->getHistoProf2(1050)->Fill(z,polarRadius,theTotalIL_TOT);
	hmgr->getHisto2(1061)->Fill(z,polarRadius);
	hmgr->getHisto2(1060)->Fill(z,polarRadius,fillValue);
	// Support
	hmgr->getHisto1(1141)->Fill(polarRadius);
	hmgr->getHistoProf1(1140)->Fill(polarRadius,theTotalIL_SUP);
	hmgr->getHisto2(1151)->Fill(z,polarRadius);
	hmgr->getHistoProf2(1150)->Fill(z,polarRadius,theTotalIL_SUP);
	hmgr->getHisto2(1161)->Fill(z,polarRadius);
	hmgr->getHisto2(1160)->Fill(z,polarRadius,iSup*fillValue);
	// Sensitive
	hmgr->getHisto1(1241)->Fill(polarRadius);
	hmgr->getHistoProf1(1240)->Fill(polarRadius,theTotalIL_SEN);
	hmgr->getHisto2(1251)->Fill(z,polarRadius);
	hmgr->getHistoProf2(1250)->Fill(z,polarRadius,theTotalIL_SEN);
	hmgr->getHisto2(1261)->Fill(z,polarRadius);
	hmgr->getHisto2(1260)->Fill(z,polarRadius,iSen*fillValue);
	// Cables
	hmgr->getHisto1(1341)->Fill(polarRadius);
	hmgr->getHistoProf1(1340)->Fill(polarRadius,theTotalIL_CAB);
	hmgr->getHisto2(1351)->Fill(z,polarRadius);
	hmgr->getHistoProf2(1350)->Fill(z,polarRadius,theTotalIL_CAB);
	hmgr->getHisto2(1361)->Fill(z,polarRadius);
	hmgr->getHisto2(1360)->Fill(z,polarRadius,iCab*fillValue);
	// Cooling
	hmgr->getHisto1(1441)->Fill(polarRadius);
	hmgr->getHistoProf1(1440)->Fill(polarRadius,theTotalIL_COL);
	hmgr->getHisto2(1451)->Fill(z,polarRadius);
	hmgr->getHistoProf2(1450)->Fill(z,polarRadius,theTotalIL_COL);
	hmgr->getHisto2(1461)->Fill(z,polarRadius);
	hmgr->getHisto2(1460)->Fill(z,polarRadius,iCol*fillValue);
	// Electronics
	hmgr->getHisto1(1541)->Fill(polarRadius);
	hmgr->getHistoProf1(1540)->Fill(polarRadius,theTotalIL_ELE);
	hmgr->getHisto2(1551)->Fill(z,polarRadius);
	hmgr->getHistoProf2(1550)->Fill(z,polarRadius,theTotalIL_ELE);
	hmgr->getHisto2(1561)->Fill(z,polarRadius);
	hmgr->getHisto2(1560)->Fill(z,polarRadius,iEle*fillValue);
	// Other
	hmgr->getHisto1(1641)->Fill(polarRadius);
	hmgr->getHistoProf1(1640)->Fill(polarRadius,theTotalIL_OTH);
	hmgr->getHisto2(1651)->Fill(z,polarRadius);
	hmgr->getHistoProf2(1650)->Fill(z,polarRadius,theTotalIL_OTH);
	hmgr->getHisto2(1661)->Fill(z,polarRadius);
	hmgr->getHisto2(1660)->Fill(z,polarRadius,iOth*fillValue);
	// Air
	hmgr->getHisto1(1741)->Fill(polarRadius);
	hmgr->getHistoProf1(1740)->Fill(polarRadius,theTotalIL_AIR);
	hmgr->getHisto2(1751)->Fill(z,polarRadius);
	hmgr->getHistoProf2(1750)->Fill(z,polarRadius,theTotalIL_AIR);
	hmgr->getHisto2(1761)->Fill(z,polarRadius);
	hmgr->getHisto2(1760)->Fill(z,polarRadius,iAir*fillValue);
	//
      }

    }
    
    // rr
  } else {
    edm::LogWarning("MaterialBudget") 
      << "MaterialBudgetTrackerHistos: This event is out of the acceptance:" 
      << "eta = "      << theData->getEta()
      << "\t phi = "   << theData->getPhi()
      << "\t x/X0 = "  << theData->getTotalMB()
      << "\t l/l0 = "  << theData->getTotalIL()
      << "\t steps = " << theData->getNumberOfSteps();
  }
}

void MaterialBudgetTrackerHistos::endOfRun()
{

  // Prefered method to include any instruction
  // once all the tracks are done

  hmgr->getHisto2(60)->Divide(hmgr->getHisto2(999));
  hmgr->getHisto2(160)->Divide(hmgr->getHisto2(999));
  hmgr->getHisto2(260)->Divide(hmgr->getHisto2(999));
  hmgr->getHisto2(360)->Divide(hmgr->getHisto2(999));
  hmgr->getHisto2(460)->Divide(hmgr->getHisto2(999));
  hmgr->getHisto2(560)->Divide(hmgr->getHisto2(999));
  hmgr->getHisto2(660)->Divide(hmgr->getHisto2(999));
  hmgr->getHisto2(760)->Divide(hmgr->getHisto2(999));
  
  hmgr->getHisto2(160)->Divide(hmgr->getHisto2(1999));
  hmgr->getHisto2(1160)->Divide(hmgr->getHisto2(1999));
  hmgr->getHisto2(1260)->Divide(hmgr->getHisto2(1999));
  hmgr->getHisto2(1360)->Divide(hmgr->getHisto2(1999));
  hmgr->getHisto2(1460)->Divide(hmgr->getHisto2(1999));
  hmgr->getHisto2(1560)->Divide(hmgr->getHisto2(1999));
  hmgr->getHisto2(1660)->Divide(hmgr->getHisto2(1999));
  hmgr->getHisto2(1760)->Divide(hmgr->getHisto2(1999));
  
  edm::LogInfo("MaterialBudget") << "MaterialBudgetTrackerHistos: Saving Histograms to: " << theFileName;
  hmgr->save( theFileName );

}
