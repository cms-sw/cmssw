#include "Validation/Geometry/interface/MaterialBudgetTrackerHistos.h"
#include "Validation/Geometry/interface/MaterialBudgetData.h"

#include "Validation/Geometry/interface/TestHistoMgr.h"

#include "SimG4Core/Notification/interface/Singleton.h"

MaterialBudgetTrackerHistos::MaterialBudgetTrackerHistos(MaterialBudgetData* data, const std::string& fileName ): MaterialBudgetFormat( data )
{
  hmgr = Singleton<TestHistoMgr>::instance();
  theFileName = fileName;
  book();

}


void MaterialBudgetTrackerHistos::book() 
{
  std::cout << "=== booking user histos ===" << std::endl;
  
  // total X0
  hmgr->addHistoProf1( new TProfile("10", "MB prof Eta;#eta;x/X_{0} ", 250, -5., 5. ) );
  hmgr->addHisto1( new TH1F("11", "Eta " , 501, -5., 5. ) );
  hmgr->addHistoProf1( new TProfile("20", "MB prof Phi;#varphi [rad];x/X_{0} ", 180, -3.1416, 3.1416 ) );
  hmgr->addHisto1( new TH1F("21", "Phi " , 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf2( new TProfile2D("30", "MB prof Eta  Phi;#eta;#varphi;x/X_{0} ", 250, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHisto2( new TH2F("31", "Eta vs Phi " , 501, -5., 5., 180, -3.1416, 3.1416 ) );
  
  // rr
  hmgr->addHistoProf1( new TProfile("40", "MB prof R;R [cm];x/X_{0} ", 200, 0., 2000. ) );
  hmgr->addHisto1( new TH1F("41", "R " , 200, 0., 2000. ) );
  
  // Support
  hmgr->addHistoProf1( new TProfile("110", "MB prof Eta [Support];#eta;x/X_{0}", 250, -5.0, 5.0 ) );
  hmgr->addHisto1( new TH1F("111", "Eta [Support]" , 501, -5., 5. ) );
  hmgr->addHistoProf1( new TProfile("120", "MB prof Phi [Support];#varphi [rad];x/X_{0}", 180, -3.1416, 3.1416 ) );
  hmgr->addHisto1( new TH1F("121", "Phi [Support]" , 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf2( new TProfile2D("130", "MB prof Eta  Phi [Support];#eta;#varphi;x/X_{0}", 250, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHisto2( new TH2F("131", "Eta vs Phi [Support]" , 501, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf1( new TProfile("140", "MB prof R [Support];R [cm];x/X_{0}", 200, 0., 2000. ) );
  hmgr->addHisto1( new TH1F("141", "R [Support]" , 200, 0., 2000. ) );
  // Sensitive
  hmgr->addHistoProf1( new TProfile("210", "MB prof Eta [Sensitive];#eta;x/X_{0}", 250, -5.0, 5.0 ) );
  hmgr->addHisto1( new TH1F("211", "Eta [Sensitive]" , 501, -5., 5. ) );
  hmgr->addHistoProf1( new TProfile("220", "MB prof Phi [Sensitive];#varphi [rad];x/X_{0}", 180, -3.1416, 3.1416 ) );
  hmgr->addHisto1( new TH1F("221", "Phi [Sensitive]" , 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf2( new TProfile2D("230", "MB prof Eta  Phi [Sensitive];#eta;#varphi;x/X_{0}", 250, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHisto2( new TH2F("231", "Eta vs Phi [Sensitive]" , 501, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf1( new TProfile("240", "MB prof R [Sensitive];R [cm];x/X_{0}", 200, 0., 2000. ) );
  hmgr->addHisto1( new TH1F("241", "R [Sensitive]" , 200, 0., 2000. ) );
  // Cables
  hmgr->addHistoProf1( new TProfile("310", "MB prof Eta [Cables];#eta;x/X_{0}", 250, -5.0, 5.0 ) );
  hmgr->addHisto1( new TH1F("311", "Eta [Cables]" , 501, -5., 5. ) );
  hmgr->addHistoProf1( new TProfile("320", "MB prof Phi [Cables];#varphi [rad];x/X_{0}", 180, -3.1416, 3.1416 ) );
  hmgr->addHisto1( new TH1F("321", "Phi [Cables]" , 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf2( new TProfile2D("330", "MB prof Eta  Phi [Cables];#eta;#varphi;x/X_{0}", 250, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHisto2( new TH2F("331", "Eta vs Phi [Cables]" , 501, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf1( new TProfile("340", "MB prof R [Cables];R [cm];x/X_{0}", 200, 0., 2000. ) );
  hmgr->addHisto1( new TH1F("341", "R [Cables]" , 200, 0., 2000. ) );
  // Cooling
  hmgr->addHistoProf1( new TProfile("410", "MB prof Eta [Cooling];#eta;x/X_{0}", 250, -5.0, 5.0 ) );
  hmgr->addHisto1( new TH1F("411", "Eta [Cooling]" , 501, -5., 5. ) );
  hmgr->addHistoProf1( new TProfile("420", "MB prof Phi [Cooling];#varphi [rad];x/X_{0}", 180, -3.1416, 3.1416 ) );
  hmgr->addHisto1( new TH1F("421", "Phi [Cooling]" , 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf2( new TProfile2D("430", "MB prof Eta  Phi [Cooling];#eta;#varphi;x/X_{0}", 250, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHisto2( new TH2F("431", "Eta vs Phi [Cooling]" , 501, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf1( new TProfile("440", "MB prof R [Cooling];R [cm];x/X_{0}", 200, 0., 2000. ) );
  hmgr->addHisto1( new TH1F("441", "R [Cooling]" , 200, 0., 2000. ) );
  // Electronics
  hmgr->addHistoProf1( new TProfile("510", "MB prof Eta [Electronics];#eta;x/X_{0}", 250, -5.0, 5.0 ) );
  hmgr->addHisto1( new TH1F("511", "Eta [Electronics]" , 501, -5., 5. ) );
  hmgr->addHistoProf1( new TProfile("520", "MB prof Phi [Electronics];#varphi [rad];x/X_{0}", 180, -3.1416, 3.1416 ) );
  hmgr->addHisto1( new TH1F("521", "Phi [Electronics]" , 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf2( new TProfile2D("530", "MB prof Eta  Phi [Electronics];#eta;#varphi;x/X_{0}", 250, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHisto2( new TH2F("531", "Eta vs Phi [Electronics]" , 501, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf1( new TProfile("540", "MB prof R [Electronics];R [cm];x/X_{0}", 200, 0., 2000. ) );
  hmgr->addHisto1( new TH1F("541", "R [Electronics]" , 200, 0., 2000. ) );
  // Other
  hmgr->addHistoProf1( new TProfile("610", "MB prof Eta [Other];#eta;x/X_{0}", 250, -5.0, 5.0 ) );
  hmgr->addHisto1( new TH1F("611", "Eta [Other]" , 501, -5., 5. ) );
  hmgr->addHistoProf1( new TProfile("620", "MB prof Phi [Other];#varphi [rad];x/X_{0}", 180, -3.1416, 3.1416 ) );
  hmgr->addHisto1( new TH1F("621", "Phi [Other]" , 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf2( new TProfile2D("630", "MB prof Eta  Phi [Other];#eta;#varphi;x/X_{0}", 250, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHisto2( new TH2F("631", "Eta vs Phi [Other]" , 501, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf1( new TProfile("640", "MB prof R [Other];R [cm];x/X_{0}", 200, 0., 2000. ) );
  hmgr->addHisto1( new TH1F("641", "R [Other]" , 200, 0., 2000. ) );
  // Air
  hmgr->addHistoProf1( new TProfile("710", "MB prof Eta [Air];#eta;x/X_{0}", 250, -5.0, 5.0 ) );
  hmgr->addHisto1( new TH1F("711", "Eta [Air]" , 501, -5., 5. ) );
  hmgr->addHistoProf1( new TProfile("720", "MB prof Phi [Air];#varphi [rad];x/X_{0}", 180, -3.1416, 3.1416 ) );
  hmgr->addHisto1( new TH1F("721", "Phi [Air]" , 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf2( new TProfile2D("730", "MB prof Eta  Phi [Air];#eta;#varphi;x/X_{0}", 250, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHisto2( new TH2F("731", "Eta vs Phi [Air]" , 501, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf1( new TProfile("740", "MB prof R [Air];R [cm];x/X_{0}", 200, 0., 2000. ) );
  hmgr->addHisto1( new TH1F("741", "R [Air]" , 200, 0., 2000. ) );
  //
  
  // rr
  
  std::cout << "=== booking user histos done ===" << std::endl;

}


void MaterialBudgetTrackerHistos::fillStartTrack()
{

}


void MaterialBudgetTrackerHistos::fillPerStep()
{

}


void MaterialBudgetTrackerHistos::fillEndTrack()
{
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
  
  // Compute the total x/X0 crossed at each step radius for each path
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
    float polarRadius = sqrt(
			     ( 0.5 * ( theData->getStepInitialX(iStep)+theData->getStepFinalX(iStep) ) )
			     * 
			     ( 0.5 * ( theData->getStepInitialX(iStep)+theData->getStepFinalX(iStep) ) )
			     +
			     ( 0.5 * ( theData->getStepInitialY(iStep)+theData->getStepFinalY(iStep) ) )
			     *
			     ( 0.5 * ( theData->getStepInitialY(iStep)+theData->getStepFinalY(iStep) ) )
			     );
    // Total
    hmgr->getHisto1(41)->Fill(polarRadius);
    hmgr->getHistoProf1(40)->Fill(polarRadius,theTotalMB_TOT);
    // Support
    hmgr->getHisto1(141)->Fill(polarRadius);
    hmgr->getHistoProf1(140)->Fill(polarRadius,theTotalMB_SUP);
    // Sensitive
    hmgr->getHisto1(241)->Fill(polarRadius);
    hmgr->getHistoProf1(240)->Fill(polarRadius,theTotalMB_SEN);
    // Cables
    hmgr->getHisto1(341)->Fill(polarRadius);
    hmgr->getHistoProf1(340)->Fill(polarRadius,theTotalMB_CAB);
    // Cooling
    hmgr->getHisto1(441)->Fill(polarRadius);
    hmgr->getHistoProf1(440)->Fill(polarRadius,theTotalMB_COL);
    // Electronics
    hmgr->getHisto1(541)->Fill(polarRadius);
    hmgr->getHistoProf1(540)->Fill(polarRadius,theTotalMB_ELE);
    // Other
    hmgr->getHisto1(641)->Fill(polarRadius);
    hmgr->getHistoProf1(640)->Fill(polarRadius,theTotalMB_OTH);
    // Air
    hmgr->getHisto1(741)->Fill(polarRadius);
    hmgr->getHistoProf1(740)->Fill(polarRadius,theTotalMB_AIR);
    //
  }
  
  // rr
}


void MaterialBudgetTrackerHistos::hend() 
{
  std::cout << "=== save user histos ===" << std::endl;
  hmgr->save( theFileName );

}

