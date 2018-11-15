#include "Validation/Geometry/interface/MaterialBudgetEcalHistos.h"
#include "Validation/Geometry/interface/MaterialBudgetData.h"


#include "CLHEP/Units/GlobalSystemOfUnits.h"

MaterialBudgetEcalHistos::MaterialBudgetEcalHistos(std::shared_ptr<MaterialBudgetData> data, 
						   std::shared_ptr<TestHistoMgr> mgr,
						   const std::string& fileName ): MaterialBudgetFormat( data ), hmgr(mgr)
{
  theFileName = fileName;
  book();
}


void MaterialBudgetEcalHistos::book() 
{
  edm::LogInfo("MaterialBudget") << "MaterialBudgetEcalHistos: Booking user histos";
  
  // total X0
  hmgr->addHistoProf1( new TProfile("10", "MB prof Eta ", 250, -5., 5. ) );
  hmgr->addHisto1( new TH1F("11", "Eta " , 501, -5., 5. ) );
  hmgr->addHistoProf1( new TProfile("20", "MB prof Phi ", 180, -3.1416, 3.1416 ) );
  hmgr->addHisto1( new TH1F("21", "Phi " , 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf2( new TProfile2D("30", "MB prof Eta  Phi ", 250, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHisto2( new TH2F("31", "Eta vs Phi " , 501, -5., 5., 180, -3.1416, 3.1416 ) );
    
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

  // ECAL specific
  hmgr->addHistoProf1( new TProfile("1001", "MB prof Eta ECAL Barrel", 340, -1.5, 1.5 ) );
  hmgr->addHistoProf1( new TProfile("1002", "MB prof Phi ECAL Barrel", 180, -3.1416, 3.1416 ) );
  hmgr->addHistoProf1( new TProfile("1003", "MB prof Phi ECAL Barrel SM", 20, 0., 20. ) );
  hmgr->addHistoProf1( new TProfile("2003", "MB prof Phi ECAL Barrel SM", 10, 0., 20. ) );
  hmgr->addHistoProf1( new TProfile("1004", "MB prof Phi ECAL Barrel SM module 1", 20, 0., 20. ) );
  hmgr->addHistoProf1( new TProfile("1005", "MB prof Phi ECAL Barrel SM module 2", 20, 0., 20. ) );
  hmgr->addHistoProf1( new TProfile("1006", "MB prof Phi ECAL Barrel SM module 3", 20, 0., 20. ) );
  hmgr->addHistoProf1( new TProfile("1007", "MB prof Phi ECAL Barrel SM module 4", 20, 0., 20. ) );

  hmgr->addHistoProf1( new TProfile("1011", "MB prof Eta ECAL Preshower +", 100, 1.65, 2.6 ) );
  hmgr->addHistoProf1( new TProfile("1012", "MB prof Phi ECAL Preshower +", 180, -3.1416, 3.1416 ) );
  
  hmgr->addHistoProf1( new TProfile("1013", "MB prof Eta ECAL Preshower -", 100, -2.6, -1.65 ) );
  hmgr->addHistoProf1( new TProfile("1014", "MB prof Phi ECAL Preshower -", 180, -3.1416, 3.1416 ) );
  
  edm::LogInfo("MaterialBudget") << "MaterialBudgetEcalHistos: booking user histos done";

}


void MaterialBudgetEcalHistos::fillStartTrack()
{

}


void MaterialBudgetEcalHistos::fillPerStep()
{

}


void MaterialBudgetEcalHistos::fillEndTrack()
{
  // Total X0
  hmgr->getHisto1(11)->Fill(theData->getEta());
  hmgr->getHisto1(21)->Fill(theData->getPhi());
  hmgr->getHisto2(31)->Fill(theData->getEta(),theData->getPhi());
  
  hmgr->getHistoProf1(10)->Fill(theData->getEta(),theData->getTotalMB());
  hmgr->getHistoProf1(20)->Fill(theData->getPhi(),theData->getTotalMB());
  hmgr->getHistoProf2(30)->Fill(theData->getEta(),theData->getPhi(),theData->getTotalMB());
  
  
  // ECAL specific
  if (fabs(theData->getEta()) <= 1.479 ) {
    static const double twenty ( 20.*degree ) ;
    const double phi ( theData->getPhi()+M_PI ) ;
    const double phiModTwenty (( phi - floor(phi/twenty)*twenty )/degree) ;
    hmgr->getHistoProf1(1001)->Fill(theData->getEta(),theData->getTotalMB());
    hmgr->getHistoProf1(1002)->Fill(theData->getPhi(),theData->getTotalMB());
    hmgr->getHistoProf1(1003)->Fill(phiModTwenty,theData->getTotalMB());
    hmgr->getHistoProf1(2003)->Fill(phiModTwenty,theData->getTotalMB());
    if (fabs(theData->getEta()) >= 0. && fabs(theData->getEta()) < 0.435 ) {
      hmgr->getHistoProf1(1004)->Fill(phiModTwenty,theData->getTotalMB());
    }
    if (fabs(theData->getEta()) >= 0.435 && fabs(theData->getEta()) < 0.783 ) {
      hmgr->getHistoProf1(1005)->Fill(phiModTwenty,theData->getTotalMB());
    }
    if (fabs(theData->getEta()) > 0.783 && fabs(theData->getEta()) <= 1.131 ) {
      hmgr->getHistoProf1(1006)->Fill(phiModTwenty,theData->getTotalMB());
    }
    if (fabs(theData->getEta()) > 1.131 && fabs(theData->getEta()) <= 1.479 ) {
      hmgr->getHistoProf1(1007)->Fill(phiModTwenty,theData->getTotalMB());
    }
  }

  if (theData->getEta() >= 1.653 && theData->getEta() <= 2.6 ) {
    hmgr->getHistoProf1(1011)->Fill(theData->getEta(),theData->getTotalMB());
    hmgr->getHistoProf1(1012)->Fill(theData->getPhi(),theData->getTotalMB());
  }

  if (theData->getEta() >= -2.6 && theData->getEta() <= -1.653 ) {
    hmgr->getHistoProf1(1013)->Fill(theData->getEta(),theData->getTotalMB());
    hmgr->getHistoProf1(1014)->Fill(theData->getPhi(),theData->getTotalMB());
  }
}


void MaterialBudgetEcalHistos::endOfRun() 
{
  edm::LogInfo("MaterialBudget") << "MaterialBudgetEcalHistos: Writing histos ROOT file to:" << theFileName;
  hmgr->save( theFileName );

}

