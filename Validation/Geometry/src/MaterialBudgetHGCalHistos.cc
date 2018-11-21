#include "Validation/Geometry/interface/MaterialBudgetHGCalHistos.h"
#include "Validation/Geometry/interface/MaterialBudgetData.h"

template <class T> const T& max ( const T& a, const T& b ) {
  return (b<a)?a:b;     // or: return comp(b,a)?a:b; for the comp version
}



MaterialBudgetHGCalHistos::MaterialBudgetHGCalHistos(std::shared_ptr<MaterialBudgetData> data,
						     std::shared_ptr<TestHistoMgr> mgr,
						     const std::string& fileName )
  : MaterialBudgetFormat( data ), 
    hmgr(mgr)
{
  theFileName = fileName;
  book();
}


void MaterialBudgetHGCalHistos::book() 
{
  std::cout << "=== booking user histos ===" << std::endl;

  // Parameters for 2D histograms
  // Make z 1mm per bin 
  int nzbin = 11000;
  float zMax = 5500.;//
  float zMin = -5500.;//
  // Make r 1cm per bin
  int nrbin = 345;
  float rMin = -50.;
  float rMax = 3400.;
  
  // total X0
  hmgr->addHistoProf1( new TProfile("10", "MB prof Eta;#eta;x/X_{0} ", 250, -5., 5. ) );
  hmgr->addHisto1( new TH1F("11", "Eta " , 250, -5., 5. ) );
  hmgr->addHistoProf1( new TProfile("20", "MB prof Phi;#varphi [rad];x/X_{0} ", 180, -3.2, 3.2 ) );
  hmgr->addHisto1( new TH1F("21", "Phi " , 180, -3.2, 3.2 ) );
  hmgr->addHistoProf2( new TProfile2D("30", "MB prof Eta  Phi;#eta;#varphi;x/X_{0} ", 250, -5., 5., 180, -3.2, 3.2) );
  hmgr->addHisto2( new TH2F("31", "Eta vs Phi " , 250, -5., 5., 180, -3.2, 3.2 ) );
  hmgr->addHistoProf1( new TProfile("40", "MB prof R;R [mm];x/X_{0} ", 300, 0., 3000. ) );
  hmgr->addHisto1( new TH1F("41", "R " , 300, 0., 3000. ) );
  hmgr->addHistoProf2( new TProfile2D("50", "MB prof sum R  z;z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("999", "Tot track length for MB", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("51", "R vs z " , nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("52", "MB ortho prof sum R  z;z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("60", "MB prof local R  z;z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("70", "MB prof local R  z;z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("72", "MB ortho prof local R  z;z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );

  // Copper
  hmgr->addHistoProf1( new TProfile("110", "MB prof Eta [Copper];#eta;x/X_{0}", 250, -5.0, 5.0 ) );
  hmgr->addHistoProf1( new TProfile("120", "MB prof Phi [Copper];#varphi [rad];x/X_{0}", 180, -3.2, 3.2 ) );
  hmgr->addHistoProf2( new TProfile2D("130", "MB prof Eta  Phi [Copper];#eta;#varphi;x/X_{0}", 250, -5., 5., 180, -3.2, 3.2 ) );
  hmgr->addHistoProf1( new TProfile("140", "MB prof R [Copper];R [mm];x/X_{0}", 300, 0., 3000. ) );
  hmgr->addHistoProf2( new TProfile2D("150", "MB prof sum R  z [Copper];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("152", "MB ortho prof sum R  z [Copper];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("160", "MB prof local R  z [Copper];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("170", "MB prof local R  z [Copper];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("172", "MB ortho prof local R  z [Copper];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );

  // H_Scintillator
  hmgr->addHistoProf1( new TProfile("210", "MB prof Eta [Scintillator];#eta;x/X_{0}", 250, -5.0, 5.0 ) );
  hmgr->addHistoProf1( new TProfile("220", "MB prof Phi [Scintillator];#varphi [rad];x/X_{0}", 180, -3.2, 3.2 ) );
  hmgr->addHistoProf2( new TProfile2D("230", "MB prof Eta  Phi [Scintillator];#eta;#varphi;x/X_{0}", 250, -5., 5., 180, -3.2, 3.2 ) );
  hmgr->addHistoProf1( new TProfile("240", "MB prof R [Scintillator];R [mm];x/X_{0}", 300, 0., 3000. ) );
  hmgr->addHistoProf2( new TProfile2D("250", "MB prof sum R  z [Scintillator];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("252", "MB ortho prof sum R  z [Scintillator];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("260", "MB prof local R  z [Scintillator];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("270", "MB prof local R  z [Scintillator];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("272", "MB ortho prof local R  z [Scintillator];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );

  // Cables
  hmgr->addHistoProf1( new TProfile("310", "MB prof Eta [Cables];#eta;x/X_{0}", 250, -5.0, 5.0 ) );
  hmgr->addHistoProf1( new TProfile("320", "MB prof Phi [Cables];#varphi [rad];x/X_{0}", 180, -3.2, 3.2 ) );
  hmgr->addHistoProf2( new TProfile2D("330", "MB prof Eta  Phi [Cables];#eta;#varphi;x/X_{0}", 250, -5., 5., 180, -3.2, 3.2 ) );
  hmgr->addHistoProf1( new TProfile("340", "MB prof R [Cables];R [mm];x/X_{0}", 300, 0., 3000. ) );
  hmgr->addHistoProf2( new TProfile2D("350", "MB prof sum R  z [Cables];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("352", "MB ortho prof sum R  z [Cables];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("360", "MB prof local R  z [Cables];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("370", "MB prof local R  z [Cables];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("372", "MB ortho prof local R  z [Cables];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );

  // M_NEMA_FR4_plate
  hmgr->addHistoProf1( new TProfile("410", "MB prof Eta [M_NEMA_FR4_plate];#eta;x/X_{0}", 250, -5.0, 5.0 ) );
  hmgr->addHistoProf1( new TProfile("420", "MB prof Phi [M_NEMA_FR4_plate];#varphi [rad];x/X_{0}", 180, -3.2, 3.2 ) );
  hmgr->addHistoProf2( new TProfile2D("430", "MB prof Eta  Phi [M_NEMA_FR4_plate];#eta;#varphi;x/X_{0}", 250, -5., 5., 180, -3.2, 3.2 ) );
  hmgr->addHistoProf1( new TProfile("440", "MB prof R [M_NEMA_FR4_plate];R [mm];x/X_{0}", 300, 0., 3000. ) );
  hmgr->addHistoProf2( new TProfile2D("450", "MB prof sum R  z [M_NEMA_FR4_plate];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("452", "MB ortho prof sum R  z [M_NEMA_FR4_plate];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("460", "MB prof local R  z [M_NEMA_FR4_plate];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("470", "MB prof local R  z [M_NEMA_FR4_plate];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("472", "MB ortho prof local R  z [M_NEMA_FR4_plate];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );

  // Silicon
  hmgr->addHistoProf1( new TProfile("510", "MB prof Eta [Silicon];#eta;x/X_{0}", 250, -5.0, 5.0 ) );
  hmgr->addHistoProf1( new TProfile("520", "MB prof Phi [Silicon];#varphi [rad];x/X_{0}", 180, -3.2, 3.2 ) );
  hmgr->addHistoProf2( new TProfile2D("530", "MB prof Eta  Phi [Silicon];#eta;#varphi;x/X_{0}", 250, -5., 5., 180, -3.2, 3.2 ) );
  hmgr->addHistoProf1( new TProfile("540", "MB prof R [Silicon];R [mm];x/X_{0}", 300, 0., 3000. ) );
  hmgr->addHistoProf2( new TProfile2D("550", "MB prof sum R  z [Silicon];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("552", "MB ortho prof sum R  z [Silicon];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("560", "MB prof local R  z [Silicon];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("570", "MB prof local R  z [Silicon];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("572", "MB ortho prof local R  z [Silicon];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );

  // Other
  hmgr->addHistoProf1( new TProfile("610", "MB prof Eta [Other];#eta;x/X_{0}", 250, -5.0, 5.0 ) );
  hmgr->addHistoProf1( new TProfile("620", "MB prof Phi [Other];#varphi [rad];x/X_{0}", 180, -3.2, 3.2 ) );
  hmgr->addHistoProf2( new TProfile2D("630", "MB prof Eta  Phi [Other];#eta;#varphi;x/X_{0}", 250, -5., 5., 180, -3.2, 3.2 ) );
  hmgr->addHistoProf1( new TProfile("640", "MB prof R [Other];R [mm];x/X_{0}", 300, 0., 3000. ) );
  hmgr->addHistoProf2( new TProfile2D("650", "MB prof sum R  z [Other];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("652", "MB ortho prof sum R  z [Other];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("660", "MB prof local R  z [Other];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("670", "MB prof local R  z [Other];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("672", "MB ortho prof local R  z [Other];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );

  // Air
  hmgr->addHistoProf1( new TProfile("710", "MB prof Eta [Air];#eta;x/X_{0}", 250, -5.0, 5.0 ) );
  hmgr->addHistoProf1( new TProfile("720", "MB prof Phi [Air];#varphi [rad];x/X_{0}", 180, -3.2, 3.2 ) );
  hmgr->addHistoProf2( new TProfile2D("730", "MB prof Eta  Phi [Air];#eta;#varphi;x/X_{0}", 250, -5., 5., 180, -3.2, 3.2 ) );
  hmgr->addHistoProf1( new TProfile("740", "MB prof R [Air];R [mm];x/X_{0}", 300, 0., 3000. ) );
  hmgr->addHistoProf2( new TProfile2D("750", "MB prof sum R  z [Air];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("752", "MB ortho prof sum R  z [Air];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("760", "MB prof local R  z [Air];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("770", "MB prof local R  z [Air];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("772", "MB ortho prof local R  z [Air];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );

  //StainlessSteel
  hmgr->addHistoProf1( new TProfile("810", "MB prof Eta [StainlessSteel];#eta;x/X_{0}", 250, -5.0, 5.0 ) );
  hmgr->addHistoProf1( new TProfile("820", "MB prof Phi [StainlessSteel];#varphi [rad];x/X_{0}", 180, -3.2, 3.2 ) );
  hmgr->addHistoProf2( new TProfile2D("830", "MB prof Eta  Phi [StainlessSteel];#eta;#varphi;x/X_{0}", 250, -5., 5., 180, -3.2, 3.2 ) );
  hmgr->addHistoProf1( new TProfile("840", "MB prof R [StainlessSteel];R [mm];x/X_{0}", 300, 0., 3000. ) );
  hmgr->addHistoProf2( new TProfile2D("850", "MB prof sum R  z [StainlessSteel];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("852", "MB ortho prof sum R  z [StainlessSteel];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("860", "MB prof local R  z [StainlessSteel];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("870", "MB prof local R  z [StainlessSteel];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("872", "MB ortho prof local R  z [StainlessSteel];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );

  //WCu
  hmgr->addHistoProf1( new TProfile("910", "MB prof Eta [WCu];#eta;x/X_{0}", 250, -5.0, 5.0 ) );
  hmgr->addHistoProf1( new TProfile("920", "MB prof Phi [WCu];#varphi [rad];x/X_{0}", 180, -3.2, 3.2 ) );
  hmgr->addHistoProf2( new TProfile2D("930", "MB prof Eta  Phi [WCu];#eta;#varphi;x/X_{0}", 250, -5., 5., 180, -3.2, 3.2 ) );
  hmgr->addHistoProf1( new TProfile("940", "MB prof R [WCu];R [mm];x/X_{0}", 300, 0., 3000. ) );
  hmgr->addHistoProf2( new TProfile2D("950", "MB prof sum R  z [WCu];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("952", "MB ortho prof sum R  z [WCu];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("960", "MB prof local R  z [WCu];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("970", "MB prof local R  z [WCu];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("972", "MB ortho prof local R  z [WCu];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );

  // Lead
  hmgr->addHistoProf1( new TProfile("1010", "MB prof Eta [Lead];#eta;x/X_{0}", 250, -5.0, 5.0 ) );
  hmgr->addHistoProf1( new TProfile("1020", "MB prof Phi [Lead];#varphi [rad];x/X_{0}", 180, -3.2, 3.2 ) );
  hmgr->addHistoProf2( new TProfile2D("1030", "MB prof Eta  Phi [Lead];#eta;#varphi;x/X_{0}", 250, -5., 5., 180, -3.2, 3.2 ) );
  hmgr->addHistoProf1( new TProfile("1040", "MB prof R [Lead];R [mm];x/X_{0}", 300, 0., 3000. ) );
  hmgr->addHistoProf2( new TProfile2D("1050", "MB prof sum R  z [Lead];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("1052", "MB ortho prof sum R  z [Lead];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("1060", "MB prof local R  z [Lead];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("1070", "MB prof local R  z [Lead];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("1072", "MB ortho prof local R  z [Lead];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );

  //=========================================================================================================
  // total Lambda0
  hmgr->addHistoProf1( new TProfile("10010", "IL prof Eta;#eta;#lambda/#lambda_{0} ", 250, -5., 5. ) );
  hmgr->addHistoProf1( new TProfile("10020", "IL prof Phi;#varphi [rad];#lambda/#lambda_{0} ", 180, -3.2, 3.2 ) );
  hmgr->addHistoProf2( new TProfile2D("10030", "IL prof Eta  Phi;#eta;#varphi;#lambda/#lambda_{0} ", 250, -5., 5., 180, -3.2, 3.2 ) );

  // rr
  hmgr->addHistoProf1( new TProfile("10040", "IL prof R;R [mm];#lambda/#lambda_{0} ", 300, 0., 3000. ) );
  hmgr->addHistoProf2( new TProfile2D("10050", "IL prof sum R  z;z [mm];R [mm];#lambda/#lambda_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("10052", "IL ortho prof sum R  z;z [mm];R [mm];#lambda/#lambda_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("1999", "Tot track length for l0", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("10060", "IL prof local R  z;z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("10070", "IL prof local R  z;z [mm];R [mm];#lambda/#lambda_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("10072", "IL ortho prof local R  z;z [mm];R [mm];#lambda/#lambda_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );

  // Copper
  hmgr->addHistoProf1( new TProfile("10110", "IL prof Eta [Copper];#eta;#lambda/#lambda_{0}", 250, -5.0, 5.0 ) );
  hmgr->addHistoProf1( new TProfile("10120", "IL prof Phi [Copper];#varphi [rad];#lambda/#lambda_{0}", 180, -3.2, 3.2 ) );
  hmgr->addHistoProf2( new TProfile2D("10130", "IL prof Eta  Phi [Copper];#eta;#varphi;#lambda/#lambda_{0}", 250, -5., 5., 180, -3.2, 3.2 ) );
  hmgr->addHistoProf1( new TProfile("10140", "IL prof R [Copper];R [mm];#lambda/#lambda_{0}", 300, 0., 3000. ) );
  hmgr->addHistoProf2( new TProfile2D("10150", "IL prof sum R  z [Copper];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("10152", "IL ortho prof sum R  z [Copper];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("10160", "IL prof local R  z [Copper];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("10170", "IL prof local R  z [Copper];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("10172", "IL ortho prof local R  z [Copper];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );

  // H_Scintillator
  hmgr->addHistoProf1( new TProfile("10210", "IL prof Eta [Scintillator];#eta;#lambda/#lambda_{0}", 250, -5.0, 5.0 ) );
  hmgr->addHistoProf1( new TProfile("10220", "IL prof Phi [Scintillator];#varphi [rad];#lambda/#lambda_{0}", 180, -3.2, 3.2 ) );
  hmgr->addHistoProf2( new TProfile2D("10230", "IL prof Eta  Phi [Scintillator];#eta;#varphi;#lambda/#lambda_{0}", 250, -5., 5., 180, -3.2, 3.2 ) );
  hmgr->addHistoProf1( new TProfile("10240", "IL prof R [Scintillator];R [mm];#lambda/#lambda_{0}", 300, 0., 3000. ) );
  hmgr->addHistoProf2( new TProfile2D("10250", "IL prof sum R  z [Scintillator];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("10252", "IL ortho prof sum R  z [Scintillator];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("10260", "IL prof local R  z [Scintillator];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("10270", "IL prof local R  z [Scintillator];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("10272", "IL ortho prof local R  z [Scintillator];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );

  // Cables
  hmgr->addHistoProf1( new TProfile("10310", "IL prof Eta [Cables];#eta;#lambda/#lambda_{0}", 250, -5.0, 5.0 ) );
  hmgr->addHistoProf1( new TProfile("10320", "IL prof Phi [Cables];#varphi [rad];#lambda/#lambda_{0}", 180, -3.2, 3.2 ) );
  hmgr->addHistoProf2( new TProfile2D("10330", "IL prof Eta  Phi [Cables];#eta;#varphi;#lambda/#lambda_{0}", 250, -5., 5., 180, -3.2, 3.2 ) );
  hmgr->addHistoProf1( new TProfile("10340", "IL prof R [Cables];R [mm];#lambda/#lambda_{0}", 300, 0., 3000. ) );
  hmgr->addHistoProf2( new TProfile2D("10350", "IL prof sum R  z [Cables];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("10352", "IL ortho prof sum R  z [Cables];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("10360", "IL prof local R  z [Cables];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("10370", "IL prof local R  z [Cables];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("10372", "IL ortho prof local R  z [Cables];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );

  // M_NEMA_FR4_plate
  hmgr->addHistoProf1( new TProfile("10410", "IL prof Eta [M_NEMA_FR4_plate];#eta;#lambda/#lambda_{0}", 250, -5.0, 5.0 ) );
  hmgr->addHistoProf1( new TProfile("10420", "IL prof Phi [M_NEMA_FR4_plate];#varphi [rad];#lambda/#lambda_{0}", 180, -3.2, 3.2 ) );
  hmgr->addHistoProf2( new TProfile2D("10430", "IL prof Eta  Phi [M_NEMA_FR4_plate];#eta;#varphi;#lambda/#lambda_{0}", 250, -5., 5., 180, -3.2, 3.2 ) );
  hmgr->addHistoProf1( new TProfile("10440", "IL prof R [M_NEMA_FR4_plate];R [mm];#lambda/#lambda_{0}", 300, 0., 3000. ) );
  hmgr->addHistoProf2( new TProfile2D("10450", "IL prof sum R  z [M_NEMA_FR4_plate];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("10452", "IL ortho prof sum R  z [M_NEMA_FR4_plate];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("10460", "IL prof local R  z [M_NEMA_FR4_plate];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("10470", "IL prof local R  z [M_NEMA_FR4_plate];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("10472", "IL ortho prof local R  z [M_NEMA_FR4_plate];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );

  // Silicon
  hmgr->addHistoProf1( new TProfile("10510", "IL prof Eta [Silicon];#eta;#lambda/#lambda_{0}", 250, -5.0, 5.0 ) );
  hmgr->addHistoProf1( new TProfile("10520", "IL prof Phi [Silicon];#varphi [rad];#lambda/#lambda_{0}", 180, -3.2, 3.2 ) );
  hmgr->addHistoProf2( new TProfile2D("10530", "IL prof Eta  Phi [Silicon];#eta;#varphi;#lambda/#lambda_{0}", 250, -5., 5., 180, -3.2, 3.2 ) );
  hmgr->addHistoProf1( new TProfile("10540", "IL prof R [Silicon];R [mm];#lambda/#lambda_{0}", 300, 0., 3000. ) );
  hmgr->addHistoProf2( new TProfile2D("10550", "IL prof sum R  z [Silicon];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("10552", "IL ortho prof sum R  z [Silicon];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("10560", "IL prof local R  z [Silicon];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("10570", "IL prof local R  z [Silicon];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("10572", "IL ortho prof local R  z [Silicon];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );

  // Other
  hmgr->addHistoProf1( new TProfile("10610", "IL prof Eta [Other];#eta;#lambda/#lambda_{0}", 250, -5.0, 5.0 ) );
  hmgr->addHistoProf1( new TProfile("10620", "IL prof Phi [Other];#varphi [rad];#lambda/#lambda_{0}", 180, -3.2, 3.2 ) );
  hmgr->addHistoProf2( new TProfile2D("10630", "IL prof Eta  Phi [Other];#eta;#varphi;#lambda/#lambda_{0}", 250, -5., 5., 180, -3.2, 3.2 ) );
  hmgr->addHistoProf1( new TProfile("10640", "IL prof R [Other];R [mm];#lambda/#lambda_{0}", 300, 0., 3000. ) );
  hmgr->addHistoProf2( new TProfile2D("10650", "IL prof sum R  z [Other];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("10652", "IL ortho prof sum R  z [Other];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("10660", "IL prof local R  z [Other];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("10670", "IL prof local R  z [Other];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("10672", "IL ortho prof local R  z [Other];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );

  // Air
  hmgr->addHistoProf1( new TProfile("10710", "IL prof Eta [Air];#eta;#lambda/#lambda_{0}", 250, -5.0, 5.0 ) );
  hmgr->addHistoProf1( new TProfile("10720", "IL prof Phi [Air];#varphi [rad];#lambda/#lambda_{0}", 180, -3.2, 3.2 ) );
  hmgr->addHistoProf2( new TProfile2D("10730", "IL prof Eta  Phi [Air];#eta;#varphi;#lambda/#lambda_{0}", 250, -5., 5., 180, -3.2, 3.2 ) );
  hmgr->addHistoProf1( new TProfile("10740", "IL prof R [Air];R [mm];#lambda/#lambda_{0}", 300, 0., 3000. ) );
  hmgr->addHistoProf2( new TProfile2D("10750", "IL prof sum R  z [Air];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("10752", "IL ortho prof sum R  z [Air];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("10760", "IL prof local R  z [Air];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("10770", "IL prof local R  z [Air];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("10772", "IL ortho prof local R  z [Air];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );

  //StainlessSteel
  hmgr->addHistoProf1( new TProfile("10810", "IL prof Eta [StainlessSteel];#eta;#lambda/#lambda_{0}", 250, -5.0, 5.0 ) );
  hmgr->addHistoProf1( new TProfile("10820", "IL prof Phi [StainlessSteel];#varphi [rad];#lambda/#lambda_{0}", 180, -3.2, 3.2 ) );
  hmgr->addHistoProf2( new TProfile2D("10830", "IL prof Eta  Phi [StainlessSteel];#eta;#varphi;#lambda/#lambda_{0}", 250, -5., 5., 180, -3.2, 3.2 ) );
  hmgr->addHistoProf1( new TProfile("10840", "IL prof R [StainlessSteel];R [mm];#lambda/#lambda_{0}", 300, 0., 3000. ) );
  hmgr->addHistoProf2( new TProfile2D("10850", "IL prof sum R  z [StainlessSteel];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("10852", "IL ortho prof sum R  z [StainlessSteel];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("10860", "IL prof local R  z [StainlessSteel];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("10870", "IL prof local R  z [StainlessSteel];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("10872", "IL ortho prof local R  z [StainlessSteel];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );

  //WCu
  hmgr->addHistoProf1( new TProfile("10910", "IL prof Eta [WCu];#eta;#lambda/#lambda_{0}", 250, -5.0, 5.0 ) );
  hmgr->addHistoProf1( new TProfile("10920", "IL prof Phi [WCu];#varphi [rad];#lambda/#lambda_{0}", 180, -3.2, 3.2 ) );
  hmgr->addHistoProf2( new TProfile2D("10930", "IL prof Eta  Phi [WCu];#eta;#varphi;#lambda/#lambda_{0}", 250, -5., 5., 180, -3.2, 3.2 ) );
  hmgr->addHistoProf1( new TProfile("10940", "IL prof R [WCu];R [mm];#lambda/#lambda_{0}", 300, 0., 3000. ) );
  hmgr->addHistoProf2( new TProfile2D("10950", "IL prof sum R  z [WCu];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("10952", "IL ortho prof sum R  z [WCu];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("10960", "IL prof local R  z [WCu];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("10970", "IL prof local R  z [WCu];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("10972", "IL ortho prof local R  z [WCu];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );

  // Lead
  hmgr->addHistoProf1( new TProfile("11010", "IL prof Eta [Lead];#eta;#lambda/#lambda_{0}", 250, -5.0, 5.0 ) );
  hmgr->addHistoProf1( new TProfile("11020", "IL prof Phi [Lead];#varphi [rad];#lambda/#lambda_{0}", 180, -3.2, 3.2 ) );
  hmgr->addHistoProf2( new TProfile2D("11030", "IL prof Eta  Phi [Lead];#eta;#varphi;#lambda/#lambda_{0}", 250, -5., 5., 180, -3.2, 3.2 ) );
  hmgr->addHistoProf1( new TProfile("11040", "IL prof R [Lead];R [mm];#lambda/#lambda_{0}", 300, 0., 3000. ) );
  hmgr->addHistoProf2( new TProfile2D("11050", "IL prof sum R  z [Lead];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("11052", "IL ortho prof sum R  z [Lead];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHisto2( new TH2F("11060", "IL prof local R  z [Lead];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("11070", "IL prof local R  z [Lead];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );
  hmgr->addHistoProf2( new TProfile2D("11072", "IL ortho prof local R  z [Lead];z [mm];R [mm];x/X_{0} ", nzbin, zMin, zMax, nrbin, rMin, rMax ) );

  
  std::cout << "=== booking user histos done ===" << std::endl;

}


void MaterialBudgetHGCalHistos::fillStartTrack()
{

}


void MaterialBudgetHGCalHistos::fillPerStep()
{

}


void MaterialBudgetHGCalHistos::fillEndTrack()
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
    
    // Copper
    hmgr->getHistoProf1(110)->Fill(theData->getEta(),theData->getCopperMB());
    hmgr->getHistoProf1(120)->Fill(theData->getPhi(),theData->getCopperMB());
    hmgr->getHistoProf2(130)->Fill(theData->getEta(),theData->getPhi(),theData->getCopperMB());
    
    // H_Scintillator
    hmgr->getHistoProf1(210)->Fill(theData->getEta(),theData->getH_ScintillatorMB());
    hmgr->getHistoProf1(220)->Fill(theData->getPhi(),theData->getH_ScintillatorMB());
    hmgr->getHistoProf2(230)->Fill(theData->getEta(),theData->getPhi(),theData->getH_ScintillatorMB());
    
    // Cables
    hmgr->getHistoProf1(310)->Fill(theData->getEta(),theData->getCablesMB());
    hmgr->getHistoProf1(320)->Fill(theData->getPhi(),theData->getCablesMB());
    hmgr->getHistoProf2(330)->Fill(theData->getEta(),theData->getPhi(),theData->getCablesMB());
    
    // M_NEMA_FR4_plate
    hmgr->getHistoProf1(410)->Fill(theData->getEta(),theData->getM_NEMA_FR4_plateMB());
    hmgr->getHistoProf1(420)->Fill(theData->getPhi(),theData->getM_NEMA_FR4_plateMB());
    hmgr->getHistoProf2(430)->Fill(theData->getEta(),theData->getPhi(),theData->getM_NEMA_FR4_plateMB());
    
    // Silicon
    hmgr->getHistoProf1(510)->Fill(theData->getEta(),theData->getSiliconMB());
    hmgr->getHistoProf1(520)->Fill(theData->getPhi(),theData->getSiliconMB());
    hmgr->getHistoProf2(530)->Fill(theData->getEta(),theData->getPhi(),theData->getSiliconMB());
    
    // Other
    hmgr->getHistoProf1(610)->Fill(theData->getEta(),theData->getOtherMB());
    hmgr->getHistoProf1(620)->Fill(theData->getPhi(),theData->getOtherMB());
    hmgr->getHistoProf2(630)->Fill(theData->getEta(),theData->getPhi(),theData->getOtherMB());
    
    // Air
    hmgr->getHistoProf1(710)->Fill(theData->getEta(),theData->getAirMB());
    hmgr->getHistoProf1(720)->Fill(theData->getPhi(),theData->getAirMB());
    hmgr->getHistoProf2(730)->Fill(theData->getEta(),theData->getPhi(),theData->getAirMB());
    
    // StainlessSteel
    hmgr->getHistoProf1(810)->Fill(theData->getEta(),theData->getStainlessSteelMB());
    hmgr->getHistoProf1(820)->Fill(theData->getPhi(),theData->getStainlessSteelMB());
    hmgr->getHistoProf2(830)->Fill(theData->getEta(),theData->getPhi(),theData->getStainlessSteelMB());

    // WCu
    hmgr->getHistoProf1(910)->Fill(theData->getEta(),theData->getWCuMB());
    hmgr->getHistoProf1(920)->Fill(theData->getPhi(),theData->getWCuMB());
    hmgr->getHistoProf2(930)->Fill(theData->getEta(),theData->getPhi(),theData->getWCuMB());

    // Lead
    hmgr->getHistoProf1(1010)->Fill(theData->getEta(),theData->getLeadMB());
    hmgr->getHistoProf1(1020)->Fill(theData->getPhi(),theData->getLeadMB());
    hmgr->getHistoProf2(1030)->Fill(theData->getEta(),theData->getPhi(),theData->getLeadMB());

    //
    // Compute the total x/X0 crossed at each step radius for each path
    //
    //
    float theTotalMB_TOT = 0.0;
    float theTotalMB_COP = 0.0;
    float theTotalMB_SCI = 0.0;
    float theTotalMB_CAB = 0.0;
    float theTotalMB_MNE = 0.0;
    float theTotalMB_NIM = 0.0;
    float theTotalMB_OTH = 0.0;
    float theTotalMB_AIR = 0.0;
    float theTotalMB_SST = 0.0;
    float theTotalMB_WCU = 0.0;
    float theTotalMB_LEA = 0.0;
    for(int iStep = 0; iStep < theData->getNumberOfSteps(); iStep++) {
      theTotalMB_TOT += theData->getStepDmb(iStep);
      theTotalMB_COP += theData->getCopperDmb(iStep);
      theTotalMB_SCI += theData->getH_ScintillatorDmb(iStep);
      theTotalMB_CAB += theData->getCablesDmb(iStep);
      theTotalMB_MNE += theData->getM_NEMA_FR4_plateDmb(iStep);
      theTotalMB_NIM += theData->getSiliconDmb(iStep);
      theTotalMB_OTH += theData->getOtherDmb(iStep);
      theTotalMB_AIR += theData->getAirDmb(iStep);        
      theTotalMB_SST += theData->getStainlessSteelDmb(iStep);        
      theTotalMB_WCU += theData->getWCuDmb(iStep);        
      theTotalMB_LEA += theData->getLeadDmb(iStep);        

      int iCop = 0;
      int iSci = 0;
      int iCab = 0;
      int iMne = 0;
      int iSil = 0;
      int iOth = 0;
      int iAir = 0;
      int iSst = 0;
      int iWcu = 0;
      int iLea = 0;
      if( theData->getCopperDmb(iStep)>0.     ) { iCop = 1; }
      if( theData->getH_ScintillatorDmb(iStep)>0.   ) { iSci = 1; }
      if( theData->getCablesDmb(iStep)>0.      ) { iCab = 1; }
      if( theData->getM_NEMA_FR4_plateDmb(iStep)>0.     ) { iMne = 1; }
      if( theData->getSiliconDmb(iStep)>0. ) { iSil = 1; }
      if( theData->getOtherDmb(iStep)>0.       ) { iOth = 1; }
      if( theData->getAirDmb(iStep)>0.         ) { iAir = 1; }
      if( theData->getStainlessSteelDmb(iStep)>0.         ) { iSst = 1; }
      if( theData->getWCuDmb(iStep)>0.         ) { iWcu = 1; }
      if( theData->getLeadDmb(iStep)>0.         ) { iLea = 1; }

      float deltaRadius = sqrt(
			       pow( theData->getStepFinalX(iStep)-theData->getStepInitialX(iStep),2 )
			       +
			       pow( theData->getStepFinalY(iStep)-theData->getStepInitialY(iStep),2 )
			       );
      float deltaz = theData->getStepFinalZ(iStep)-theData->getStepInitialZ(iStep) ;
      
      float deltaeta = theData->getStepFinalEta(iStep)-theData->getStepInitialEta(iStep) ;

      // float deltaphi = theData->getStepFinalPhi(iStep)-theData->getStepInitialPhi(iStep) ;

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

	float subdeltaeta = deltaeta/nSubStep;
	float eta = theData->getStepInitialEta(iStep) + iSubStep*subdeltaeta;

	// float subdeltaphi = deltaphi/nSubStep;
	// float phi = theData->getStepInitialPhi(iStep) + iSubStep*subdeltaphi;

	float subdelta = sqrt(
			      pow ( subdeltaRadius,2 ) + pow( subdeltaz,2 )
			      );

	float fillValue=subdelta/x0;

	float costhetacorrection = cos( 2 * atan(exp(-fabs(eta))) );
	// Average length
	hmgr->getHisto2(999)->Fill(z,polarRadius,subdelta);
	// Total
	hmgr->getHisto1(41)->Fill(polarRadius);
	hmgr->getHistoProf1(40)->Fill(polarRadius,theTotalMB_TOT);
	hmgr->getHisto2(51)->Fill(z,polarRadius);
	hmgr->getHistoProf2(50)->Fill(z,polarRadius,theTotalMB_TOT);
	hmgr->getHistoProf2(52)->Fill(z,polarRadius, theTotalMB_TOT * costhetacorrection );
	hmgr->getHisto2(60)->Fill(z,polarRadius,fillValue);
	hmgr->getHistoProf2(70)->Fill(z,polarRadius,fillValue);
	hmgr->getHistoProf2(72)->Fill(z,polarRadius, fillValue * costhetacorrection );

	// Copper
	hmgr->getHistoProf1(140)->Fill(polarRadius,theTotalMB_COP);
	hmgr->getHistoProf2(150)->Fill(z,polarRadius,theTotalMB_COP);
	hmgr->getHistoProf2(152)->Fill(z,polarRadius,theTotalMB_COP * costhetacorrection );
	hmgr->getHisto2(160)->Fill(z,polarRadius,iCop*fillValue);
	hmgr->getHistoProf2(170)->Fill(z,polarRadius,iCop*fillValue);
	hmgr->getHistoProf2(172)->Fill(z,polarRadius, iCop * fillValue * costhetacorrection );

	// H_Scintillator
	hmgr->getHistoProf1(240)->Fill(polarRadius,theTotalMB_SCI);
	hmgr->getHistoProf2(250)->Fill(z,polarRadius,theTotalMB_SCI);
	hmgr->getHistoProf2(252)->Fill(z,polarRadius,theTotalMB_SCI * costhetacorrection );
	hmgr->getHisto2(260)->Fill(z,polarRadius,iSci*fillValue);
	hmgr->getHistoProf2(270)->Fill(z,polarRadius,iSci*fillValue);
	hmgr->getHistoProf2(272)->Fill(z,polarRadius, iSci * fillValue * costhetacorrection );

	// Cables
	hmgr->getHistoProf1(340)->Fill(polarRadius,theTotalMB_CAB);
	hmgr->getHistoProf2(350)->Fill(z,polarRadius,theTotalMB_CAB);
	hmgr->getHistoProf2(352)->Fill(z,polarRadius,theTotalMB_CAB * costhetacorrection );
	hmgr->getHisto2(360)->Fill(z,polarRadius,iCab*fillValue);
	hmgr->getHistoProf2(370)->Fill(z,polarRadius,iCab*fillValue);
	hmgr->getHistoProf2(372)->Fill(z,polarRadius, iCab * fillValue * costhetacorrection );

	// M_NEMA_FR4_plate
	hmgr->getHistoProf1(440)->Fill(polarRadius,theTotalMB_MNE);
	hmgr->getHistoProf2(450)->Fill(z,polarRadius,theTotalMB_MNE);
	hmgr->getHistoProf2(452)->Fill(z,polarRadius,theTotalMB_MNE * costhetacorrection );
	hmgr->getHisto2(460)->Fill(z,polarRadius,iMne*fillValue);
	hmgr->getHistoProf2(470)->Fill(z,polarRadius,iMne*fillValue);
	hmgr->getHistoProf2(472)->Fill(z,polarRadius, iMne * fillValue * costhetacorrection );

	// Silicon
	hmgr->getHistoProf1(540)->Fill(polarRadius,theTotalMB_NIM);
	hmgr->getHistoProf2(550)->Fill(z,polarRadius,theTotalMB_NIM);
	hmgr->getHistoProf2(552)->Fill(z,polarRadius,theTotalMB_NIM * costhetacorrection );
	hmgr->getHisto2(560)->Fill(z,polarRadius,iSil*fillValue);
	hmgr->getHistoProf2(570)->Fill(z,polarRadius,iSil*fillValue);
	hmgr->getHistoProf2(572)->Fill(z,polarRadius, iSil * fillValue * costhetacorrection );

	// Other
	hmgr->getHistoProf1(640)->Fill(polarRadius,theTotalMB_OTH);
	hmgr->getHistoProf2(650)->Fill(z,polarRadius,theTotalMB_OTH);
	hmgr->getHistoProf2(652)->Fill(z,polarRadius,theTotalMB_OTH * costhetacorrection );
	hmgr->getHisto2(660)->Fill(z,polarRadius,iOth*fillValue);
	hmgr->getHistoProf2(670)->Fill(z,polarRadius,iOth*fillValue);
	hmgr->getHistoProf2(672)->Fill(z,polarRadius, iOth * fillValue * costhetacorrection );

	// Air
	hmgr->getHistoProf1(740)->Fill(polarRadius,theTotalMB_AIR);
	hmgr->getHistoProf2(750)->Fill(z,polarRadius,theTotalMB_AIR);
	hmgr->getHistoProf2(752)->Fill(z,polarRadius,theTotalMB_AIR * costhetacorrection );
	hmgr->getHisto2(760)->Fill(z,polarRadius,iAir*fillValue);
	hmgr->getHistoProf2(770)->Fill(z,polarRadius,iAir*fillValue);
	hmgr->getHistoProf2(772)->Fill(z,polarRadius, iAir * fillValue * costhetacorrection );

	// StainlessSteel
	hmgr->getHistoProf1(840)->Fill(polarRadius,theTotalMB_SST);
	hmgr->getHistoProf2(850)->Fill(z,polarRadius,theTotalMB_SST);
	hmgr->getHistoProf2(852)->Fill(z,polarRadius,theTotalMB_SST * costhetacorrection );
	hmgr->getHisto2(860)->Fill(z,polarRadius,iSst*fillValue);
	hmgr->getHistoProf2(870)->Fill(z,polarRadius,iSst*fillValue);
	hmgr->getHistoProf2(872)->Fill(z,polarRadius, iSst * fillValue * costhetacorrection );

	// WCu
	hmgr->getHistoProf1(940)->Fill(polarRadius,theTotalMB_WCU);
	hmgr->getHistoProf2(950)->Fill(z,polarRadius,theTotalMB_WCU);
	hmgr->getHistoProf2(952)->Fill(z,polarRadius,theTotalMB_WCU * costhetacorrection );
	hmgr->getHisto2(960)->Fill(z,polarRadius,iWcu*fillValue);
	hmgr->getHistoProf2(970)->Fill(z,polarRadius,iWcu*fillValue);
	hmgr->getHistoProf2(972)->Fill(z,polarRadius, iWcu * fillValue * costhetacorrection );

	// Lead
	hmgr->getHistoProf1(1040)->Fill(polarRadius,theTotalMB_LEA);
	hmgr->getHistoProf2(1050)->Fill(z,polarRadius,theTotalMB_LEA);
	hmgr->getHistoProf2(1052)->Fill(z,polarRadius,theTotalMB_LEA * costhetacorrection );
	hmgr->getHisto2(1060)->Fill(z,polarRadius,iLea*fillValue);
	hmgr->getHistoProf2(1070)->Fill(z,polarRadius,iLea*fillValue);
	hmgr->getHistoProf2(1072)->Fill(z,polarRadius, iLea * fillValue  * costhetacorrection );

      }
    }
 
    //============================================================================================
    // Total Lambda0
    hmgr->getHistoProf1(10010)->Fill(theData->getEta(),theData->getTotalIL());
    hmgr->getHistoProf1(10020)->Fill(theData->getPhi(),theData->getTotalIL());
    hmgr->getHistoProf2(10030)->Fill(theData->getEta(),theData->getPhi(),theData->getTotalIL());
    
    // Copper
    hmgr->getHistoProf1(10110)->Fill(theData->getEta(),theData->getCopperIL());
    hmgr->getHistoProf1(10120)->Fill(theData->getPhi(),theData->getCopperIL());
    hmgr->getHistoProf2(10130)->Fill(theData->getEta(),theData->getPhi(),theData->getCopperIL());
    
    // H_Scintillator
    hmgr->getHistoProf1(10210)->Fill(theData->getEta(),theData->getH_ScintillatorIL());
    hmgr->getHistoProf1(10220)->Fill(theData->getPhi(),theData->getH_ScintillatorIL());
    hmgr->getHistoProf2(10230)->Fill(theData->getEta(),theData->getPhi(),theData->getH_ScintillatorIL());
    
    // Cables
    hmgr->getHistoProf1(10310)->Fill(theData->getEta(),theData->getCablesIL());
    hmgr->getHistoProf1(10320)->Fill(theData->getPhi(),theData->getCablesIL());
    hmgr->getHistoProf2(10330)->Fill(theData->getEta(),theData->getPhi(),theData->getCablesIL());
    
    // M_NEMA_FR4_plate
    hmgr->getHistoProf1(10410)->Fill(theData->getEta(),theData->getM_NEMA_FR4_plateIL());
    hmgr->getHistoProf1(10420)->Fill(theData->getPhi(),theData->getM_NEMA_FR4_plateIL());
    hmgr->getHistoProf2(10430)->Fill(theData->getEta(),theData->getPhi(),theData->getM_NEMA_FR4_plateIL());
    
    // Silicon
    hmgr->getHistoProf1(10510)->Fill(theData->getEta(),theData->getSiliconIL());
    hmgr->getHistoProf1(10520)->Fill(theData->getPhi(),theData->getSiliconIL());
    hmgr->getHistoProf2(10530)->Fill(theData->getEta(),theData->getPhi(),theData->getSiliconIL());
    
    // Other
    hmgr->getHistoProf1(10610)->Fill(theData->getEta(),theData->getOtherIL());
    hmgr->getHistoProf1(10620)->Fill(theData->getPhi(),theData->getOtherIL());
    hmgr->getHistoProf2(10630)->Fill(theData->getEta(),theData->getPhi(),theData->getOtherIL());
    
    // Air
    hmgr->getHistoProf1(10710)->Fill(theData->getEta(),theData->getAirIL());
    hmgr->getHistoProf1(10720)->Fill(theData->getPhi(),theData->getAirIL());
    hmgr->getHistoProf2(10730)->Fill(theData->getEta(),theData->getPhi(),theData->getAirIL());
    
    // StainlessSteel
    hmgr->getHistoProf1(10810)->Fill(theData->getEta(),theData->getStainlessSteelIL());
    hmgr->getHistoProf1(10820)->Fill(theData->getPhi(),theData->getStainlessSteelIL());
    hmgr->getHistoProf2(10830)->Fill(theData->getEta(),theData->getPhi(),theData->getStainlessSteelIL());

    // WCu
    hmgr->getHistoProf1(10910)->Fill(theData->getEta(),theData->getWCuIL());
    hmgr->getHistoProf1(10920)->Fill(theData->getPhi(),theData->getWCuIL());
    hmgr->getHistoProf2(10930)->Fill(theData->getEta(),theData->getPhi(),theData->getWCuIL());

    // Lead
    hmgr->getHistoProf1(11010)->Fill(theData->getEta(),theData->getLeadIL());
    hmgr->getHistoProf1(11020)->Fill(theData->getPhi(),theData->getLeadIL());
    hmgr->getHistoProf2(11030)->Fill(theData->getEta(),theData->getPhi(),theData->getLeadIL());

    // Compute the total l/l0 crossed at each step radius for each path
    float theTotalIL_TOT = 0.0;
    float theTotalIL_COP = 0.0;
    float theTotalIL_SCI = 0.0;
    float theTotalIL_CAB = 0.0;
    float theTotalIL_MNE = 0.0;
    float theTotalIL_NIM = 0.0;
    float theTotalIL_OTH = 0.0;
    float theTotalIL_AIR = 0.0;
    float theTotalIL_SST = 0.0;
    float theTotalIL_WCU = 0.0;
    float theTotalIL_LEA = 0.0;
    for(int iStep = 0; iStep < theData->getNumberOfSteps(); iStep++) {
      theTotalIL_TOT += theData->getStepDil(iStep);
      theTotalIL_COP += theData->getCopperDil(iStep);
      theTotalIL_SCI += theData->getH_ScintillatorDil(iStep);
      theTotalIL_CAB += theData->getCablesDil(iStep);
      theTotalIL_MNE += theData->getM_NEMA_FR4_plateDil(iStep);
      theTotalIL_NIM += theData->getSiliconDil(iStep);
      theTotalIL_OTH += theData->getOtherDil(iStep);
      theTotalIL_AIR += theData->getAirDil(iStep);
      theTotalIL_SST += theData->getStainlessSteelDil(iStep);
      theTotalIL_WCU += theData->getWCuDil(iStep);
      theTotalIL_LEA += theData->getLeadDil(iStep);

      int iCop = 0;
      int iSci = 0;
      int iCab = 0;
      int iMne = 0;
      int iSil = 0;
      int iOth = 0;
      int iAir = 0;
      int iSst = 0;
      int iWcu = 0;
      int iLea = 0;

      if( theData->getCopperDil(iStep)>0.     ) { iCop = 1; }
      if( theData->getH_ScintillatorDil(iStep)>0.   ) { iSci = 1; }
      if( theData->getCablesDil(iStep)>0.      ) { iCab = 1; }
      if( theData->getM_NEMA_FR4_plateDil(iStep)>0.     ) { iMne = 1; }
      if( theData->getSiliconDil(iStep)>0. ) { iSil = 1; }
      if( theData->getOtherDil(iStep)>0.       ) { iOth = 1; }
      if( theData->getAirDil(iStep)>0.         ) { iAir = 1; }
      if( theData->getStainlessSteelDil(iStep)>0.         ) { iSst = 1; }
      if( theData->getWCuDil(iStep)>0.         ) { iWcu = 1; }
      if( theData->getLeadDil(iStep)>0.         ) { iLea = 1; }

      float deltaRadius = sqrt(
			       pow( theData->getStepFinalX(iStep)-theData->getStepInitialX(iStep),2 )
			       +
			       pow( theData->getStepFinalY(iStep)-theData->getStepInitialY(iStep),2 )
			       );
      float deltaz = theData->getStepFinalZ(iStep)-theData->getStepInitialZ(iStep) ;
      
      float deltaeta = theData->getStepFinalEta(iStep)-theData->getStepInitialEta(iStep) ;

      // float deltaphi = theData->getStepFinalPhi(iStep)-theData->getStepInitialPhi(iStep) ;

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

	float subdeltaeta = deltaeta/nSubStep;
	float eta = theData->getStepInitialEta(iStep) + iSubStep*subdeltaeta;

	// float subdeltaphi = deltaphi/nSubStep;
	// float phi = theData->getStepInitialPhi(iStep) + iSubStep*subdeltaphi;

	float subdelta = sqrt(
			      pow ( subdeltaRadius,2 ) + pow( subdeltaz,2 )
			      );

	float fillValue=subdelta/il;

	float costhetacorrection = cos( 2 * atan(exp(-fabs(eta))) );
	// Average length
	hmgr->getHisto2(1999)->Fill(z,polarRadius,subdelta);
	// Total
	hmgr->getHistoProf1(10040)->Fill(polarRadius,theTotalIL_TOT);
	hmgr->getHistoProf2(10050)->Fill(z,polarRadius,theTotalIL_TOT);
	hmgr->getHistoProf2(10052)->Fill(z,polarRadius,theTotalIL_TOT * costhetacorrection );
	hmgr->getHisto2(10060)->Fill(z,polarRadius,fillValue);
	hmgr->getHistoProf2(10070)->Fill(z,polarRadius,fillValue);
	hmgr->getHistoProf2(10072)->Fill(z,polarRadius, fillValue * costhetacorrection );

	// Copper
	hmgr->getHistoProf1(10140)->Fill(polarRadius,theTotalIL_COP);
	hmgr->getHistoProf2(10150)->Fill(z,polarRadius,theTotalIL_COP);
	hmgr->getHistoProf2(10152)->Fill(z,polarRadius,theTotalIL_COP * costhetacorrection );
	hmgr->getHisto2(10160)->Fill(z,polarRadius,iCop*fillValue);
	hmgr->getHistoProf2(10170)->Fill(z,polarRadius,iCop*fillValue);
	hmgr->getHistoProf2(10172)->Fill(z,polarRadius, iCop * fillValue * costhetacorrection );

	// H_Scintillator
	hmgr->getHistoProf1(10240)->Fill(polarRadius,theTotalIL_SCI);
	hmgr->getHistoProf2(10250)->Fill(z,polarRadius,theTotalIL_SCI);
	hmgr->getHistoProf2(10252)->Fill(z,polarRadius,theTotalIL_SCI * costhetacorrection );
	hmgr->getHisto2(10260)->Fill(z,polarRadius,iSci*fillValue);
	hmgr->getHistoProf2(10270)->Fill(z,polarRadius,iSci*fillValue);
	hmgr->getHistoProf2(10272)->Fill(z,polarRadius, iSci * fillValue * costhetacorrection );

	// Cables
	hmgr->getHistoProf1(10340)->Fill(polarRadius,theTotalIL_CAB);
	hmgr->getHistoProf2(10350)->Fill(z,polarRadius,theTotalIL_CAB);
	hmgr->getHistoProf2(10352)->Fill(z,polarRadius,theTotalIL_CAB * costhetacorrection );
	hmgr->getHisto2(10360)->Fill(z,polarRadius,iCab*fillValue);
	hmgr->getHistoProf2(10370)->Fill(z,polarRadius,iCab*fillValue);
	hmgr->getHistoProf2(10372)->Fill(z,polarRadius, iCab * fillValue * costhetacorrection );

	// M_NEMA_FR4_plate
	hmgr->getHistoProf1(10440)->Fill(polarRadius,theTotalIL_MNE);
	hmgr->getHistoProf2(10450)->Fill(z,polarRadius,theTotalIL_MNE);
	hmgr->getHistoProf2(10452)->Fill(z,polarRadius,theTotalIL_MNE * costhetacorrection );
	hmgr->getHisto2(10460)->Fill(z,polarRadius,iMne*fillValue);
	hmgr->getHistoProf2(10470)->Fill(z,polarRadius,iMne*fillValue);
	hmgr->getHistoProf2(10472)->Fill(z,polarRadius, iMne * fillValue * costhetacorrection );

	// Silicon
	hmgr->getHistoProf1(10540)->Fill(polarRadius,theTotalIL_NIM);
	hmgr->getHistoProf2(10550)->Fill(z,polarRadius,theTotalIL_NIM);
	hmgr->getHistoProf2(10552)->Fill(z,polarRadius,theTotalIL_NIM * costhetacorrection );
	hmgr->getHisto2(10560)->Fill(z,polarRadius,iSil*fillValue);
	hmgr->getHistoProf2(10570)->Fill(z,polarRadius,iSil*fillValue);
	hmgr->getHistoProf2(10572)->Fill(z,polarRadius, iSil * fillValue * costhetacorrection );

	// Other
	hmgr->getHistoProf1(10640)->Fill(polarRadius,theTotalIL_OTH);
	hmgr->getHistoProf2(10650)->Fill(z,polarRadius,theTotalIL_OTH);
	hmgr->getHistoProf2(10652)->Fill(z,polarRadius,theTotalIL_OTH * costhetacorrection );
	hmgr->getHisto2(10660)->Fill(z,polarRadius,iOth*fillValue);
	hmgr->getHistoProf2(10670)->Fill(z,polarRadius,iOth*fillValue);
	hmgr->getHistoProf2(10672)->Fill(z,polarRadius, iOth * fillValue * costhetacorrection );

	// Air
	hmgr->getHistoProf1(10740)->Fill(polarRadius,theTotalIL_AIR);
	hmgr->getHistoProf2(10750)->Fill(z,polarRadius,theTotalIL_AIR);
	hmgr->getHistoProf2(10752)->Fill(z,polarRadius,theTotalIL_AIR * costhetacorrection );
	hmgr->getHisto2(10760)->Fill(z,polarRadius,iAir*fillValue);
	hmgr->getHistoProf2(10770)->Fill(z,polarRadius,iAir*fillValue);
	hmgr->getHistoProf2(10772)->Fill(z,polarRadius, iAir * fillValue * costhetacorrection );

	// StainlessSteel
	hmgr->getHistoProf1(10840)->Fill(polarRadius,theTotalIL_SST);
	hmgr->getHistoProf2(10850)->Fill(z,polarRadius,theTotalIL_SST);
	hmgr->getHistoProf2(10852)->Fill(z,polarRadius,theTotalIL_SST * costhetacorrection );
	hmgr->getHisto2(10860)->Fill(z,polarRadius,iSst*fillValue);
	hmgr->getHistoProf2(10870)->Fill(z,polarRadius,iSst*fillValue);
	hmgr->getHistoProf2(10872)->Fill(z,polarRadius, iSst * fillValue * costhetacorrection );

	// WCu
	hmgr->getHistoProf1(10940)->Fill(polarRadius,theTotalIL_WCU);
	hmgr->getHistoProf2(10950)->Fill(z,polarRadius,theTotalIL_WCU);
	hmgr->getHistoProf2(10952)->Fill(z,polarRadius,theTotalIL_WCU * costhetacorrection );
	hmgr->getHisto2(10960)->Fill(z,polarRadius,iWcu*fillValue);
	hmgr->getHistoProf2(10970)->Fill(z,polarRadius,iWcu*fillValue);
	hmgr->getHistoProf2(10972)->Fill(z,polarRadius, iWcu * fillValue * costhetacorrection );

	// Lead
	hmgr->getHistoProf1(11040)->Fill(polarRadius,theTotalIL_LEA);
	hmgr->getHistoProf2(11050)->Fill(z,polarRadius,theTotalIL_LEA);
	hmgr->getHistoProf2(11052)->Fill(z,polarRadius,theTotalIL_LEA * costhetacorrection );
	hmgr->getHisto2(11060)->Fill(z,polarRadius,iLea*fillValue);
	hmgr->getHistoProf2(11070)->Fill(z,polarRadius,iLea*fillValue);
	hmgr->getHistoProf2(11072)->Fill(z,polarRadius, iLea * fillValue * costhetacorrection );

      }

    }
    
    // rr
  } else {
    std::cout << "*** WARNING This event is out of the acceptance *** " << std::endl;
    std::cout << "eta = "      << theData->getEta()
    	      << "\t phi = "   << theData->getPhi()
    	      << "\t x/X0 = "  << theData->getTotalMB()
    	      << "\t l/l0 = "  << theData->getTotalIL()
    	      << "\t steps = " << theData->getNumberOfSteps()
    	      << std::endl;
    std::cout << "***" << std::endl;
  }

 
}

void MaterialBudgetHGCalHistos::endOfRun() 
{

  hmgr->getHisto2(60)->Divide(hmgr->getHisto2(999));
  hmgr->getHisto2(160)->Divide(hmgr->getHisto2(999));
  hmgr->getHisto2(260)->Divide(hmgr->getHisto2(999));
  hmgr->getHisto2(360)->Divide(hmgr->getHisto2(999));
  hmgr->getHisto2(460)->Divide(hmgr->getHisto2(999));
  hmgr->getHisto2(560)->Divide(hmgr->getHisto2(999));
  hmgr->getHisto2(660)->Divide(hmgr->getHisto2(999));
  hmgr->getHisto2(760)->Divide(hmgr->getHisto2(999));
  hmgr->getHisto2(860)->Divide(hmgr->getHisto2(999));
  hmgr->getHisto2(960)->Divide(hmgr->getHisto2(999));
  hmgr->getHisto2(1060)->Divide(hmgr->getHisto2(999));

  hmgr->getHisto2(10060)->Divide(hmgr->getHisto2(1999));
  hmgr->getHisto2(10160)->Divide(hmgr->getHisto2(1999));
  hmgr->getHisto2(10260)->Divide(hmgr->getHisto2(1999));
  hmgr->getHisto2(10360)->Divide(hmgr->getHisto2(1999));
  hmgr->getHisto2(10460)->Divide(hmgr->getHisto2(1999));
  hmgr->getHisto2(10560)->Divide(hmgr->getHisto2(1999));
  hmgr->getHisto2(10660)->Divide(hmgr->getHisto2(1999));
  hmgr->getHisto2(10760)->Divide(hmgr->getHisto2(1999));
  hmgr->getHisto2(10860)->Divide(hmgr->getHisto2(1999));
  hmgr->getHisto2(10960)->Divide(hmgr->getHisto2(1999));
  hmgr->getHisto2(11060)->Divide(hmgr->getHisto2(1999));

  std::cout << "=== save user histos ===" << std::endl;
  hmgr->save( theFileName );

}


