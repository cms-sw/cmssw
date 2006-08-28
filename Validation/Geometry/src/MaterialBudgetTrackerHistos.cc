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
  hmgr->addHistoProf1( new TProfile("10", "MB prof Eta ", 250, -5., 5. ) );
  hmgr->addHisto1( new TH1F("11", "Eta " , 501, -5., 5. ) );
  hmgr->addHistoProf1( new TProfile("20", "MB prof Phi ", 180, -3.1416, 3.1416 ) );
  hmgr->addHisto1( new TH1F("21", "Phi " , 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf2( new TProfile2D("30", "MB prof Eta  Phi ", 250, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHisto2( new TH2F("31", "Eta vs Phi " , 501, -5., 5., 180, -3.1416, 3.1416 ) );
  
  // rr
  
  // Support
  hmgr->addHistoProf1( new TProfile("110", "MB prof Eta [Support]", 250, -5.0, 5.0 ) );
  hmgr->addHisto1( new TH1F("111", "Eta [Support]" , 501, -5., 5. ) );
  hmgr->addHistoProf1( new TProfile("120", "MB prof Phi [Support]", 180, -3.1416, 3.1416 ) );
  hmgr->addHisto1( new TH1F("121", "Phi [Support]" , 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf2( new TProfile2D("130", "MB prof Eta  Phi [Support]", 250, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHisto2( new TH2F("131", "Eta vs Phi [Support]" , 501, -5., 5., 180, -3.1416, 3.1416 ) );
  // Sensitive
  hmgr->addHistoProf1( new TProfile("210", "MB prof Eta [Sensitive]", 250, -5.0, 5.0 ) );
  hmgr->addHisto1( new TH1F("211", "Eta [Sensitive]" , 501, -5., 5. ) );
  hmgr->addHistoProf1( new TProfile("220", "MB prof Phi [Sensitive]", 180, -3.1416, 3.1416 ) );
  hmgr->addHisto1( new TH1F("221", "Phi [Sensitive]" , 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf2( new TProfile2D("230", "MB prof Eta  Phi [Sensitive]", 250, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHisto2( new TH2F("231", "Eta vs Phi [Sensitive]" , 501, -5., 5., 180, -3.1416, 3.1416 ) );
  // Cables
  hmgr->addHistoProf1( new TProfile("310", "MB prof Eta [Cables]", 250, -5.0, 5.0 ) );
  hmgr->addHisto1( new TH1F("311", "Eta [Cables]" , 501, -5., 5. ) );
  hmgr->addHistoProf1( new TProfile("320", "MB prof Phi [Cables]", 180, -3.1416, 3.1416 ) );
  hmgr->addHisto1( new TH1F("321", "Phi [Cables]" , 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf2( new TProfile2D("330", "MB prof Eta  Phi [Cables]", 250, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHisto2( new TH2F("331", "Eta vs Phi [Cables]" , 501, -5., 5., 180, -3.1416, 3.1416 ) );
  // Cooling
  hmgr->addHistoProf1( new TProfile("410", "MB prof Eta [Cooling]", 250, -5.0, 5.0 ) );
  hmgr->addHisto1( new TH1F("411", "Eta [Cooling]" , 501, -5., 5. ) );
  hmgr->addHistoProf1( new TProfile("420", "MB prof Phi [Cooling]", 180, -3.1416, 3.1416 ) );
  hmgr->addHisto1( new TH1F("421", "Phi [Cooling]" , 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf2( new TProfile2D("430", "MB prof Eta  Phi [Cooling]", 250, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHisto2( new TH2F("431", "Eta vs Phi [Cooling]" , 501, -5., 5., 180, -3.1416, 3.1416 ) );
  // Electronics
  hmgr->addHistoProf1( new TProfile("510", "MB prof Eta [Electronics]", 250, -5.0, 5.0 ) );
  hmgr->addHisto1( new TH1F("511", "Eta [Electronics]" , 501, -5., 5. ) );
  hmgr->addHistoProf1( new TProfile("520", "MB prof Phi [Electronics]", 180, -3.1416, 3.1416 ) );
  hmgr->addHisto1( new TH1F("521", "Phi [Electronics]" , 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf2( new TProfile2D("530", "MB prof Eta  Phi [Electronics]", 250, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHisto2( new TH2F("531", "Eta vs Phi [Electronics]" , 501, -5., 5., 180, -3.1416, 3.1416 ) );
  // Other
  hmgr->addHistoProf1( new TProfile("610", "MB prof Eta [Other]", 250, -5.0, 5.0 ) );
  hmgr->addHisto1( new TH1F("611", "Eta [Other]" , 501, -5., 5. ) );
  hmgr->addHistoProf1( new TProfile("620", "MB prof Phi [Other]", 180, -3.1416, 3.1416 ) );
  hmgr->addHisto1( new TH1F("621", "Phi [Other]" , 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf2( new TProfile2D("630", "MB prof Eta  Phi [Other]", 250, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHisto2( new TH2F("631", "Eta vs Phi [Other]" , 501, -5., 5., 180, -3.1416, 3.1416 ) );
  // Air
  hmgr->addHistoProf1( new TProfile("710", "MB prof Eta [Air]", 250, -5.0, 5.0 ) );
  hmgr->addHisto1( new TH1F("711", "Eta [Air]" , 501, -5., 5. ) );
  hmgr->addHistoProf1( new TProfile("720", "MB prof Phi [Air]", 180, -3.1416, 3.1416 ) );
  hmgr->addHisto1( new TH1F("721", "Phi [Air]" , 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf2( new TProfile2D("730", "MB prof Eta  Phi [Air]", 250, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHisto2( new TH2F("731", "Eta vs Phi [Air]" , 501, -5., 5., 180, -3.1416, 3.1416 ) );
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
  
  // rr
}


void MaterialBudgetTrackerHistos::hend() 
{
  std::cout << "=== save user histos ===" << std::endl;
  hmgr->save( theFileName );

}

