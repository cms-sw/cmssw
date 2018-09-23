#include "Validation/Geometry/interface/MaterialBudgetHistos.h"
#include "Validation/Geometry/interface/MaterialBudgetData.h"


MaterialBudgetHistos::MaterialBudgetHistos(std::shared_ptr<MaterialBudgetData> data,
					   std::shared_ptr<TestHistoMgr> mgr,
					   const std::string& fileName )
  : MaterialBudgetFormat( data ), 
    hmgr(mgr)
{
  theFileName = fileName;
  book();
}


void MaterialBudgetHistos::book() 
{
  edm::LogInfo("MaterialBudget") << " MaterialBudgetHistos: Booking Histos";
  hmgr->addHistoProf1( new TProfile("10", "MB prof Eta ", 250, -5., 5. ) );
  hmgr->addHisto1( new TH1F("11", "Eta " , 501, -5., 5. ) );
  hmgr->addHistoProf1( new TProfile("20", "MB prof Phi ", 180, -3.1416, 3.1416 ) );
  hmgr->addHisto1( new TH1F("21", "Phi " , 360, -3.1416, 3.1416 ) );
  hmgr->addHistoProf2( new TProfile2D("30", "MB prof Eta  Phi ", 250, -5., 5., 180, -3.1416, 3.1416 ) );
  hmgr->addHisto2( new TH2F("31", "Eta vs Phi " , 501, -5., 5., 180, -3.1416, 3.1416 ) );
}


void MaterialBudgetHistos::fillStartTrack()
{

}


void MaterialBudgetHistos::fillPerStep()
{

}


void MaterialBudgetHistos::fillEndTrack()
{
  hmgr->getHisto1(11)->Fill(theData->getEta());
  hmgr->getHisto1(21)->Fill(theData->getPhi());
  hmgr->getHisto2(31)->Fill(theData->getEta(),theData->getPhi());

  hmgr->getHistoProf1(10)->Fill(theData->getEta(),theData->getTotalMB());
  hmgr->getHistoProf1(20)->Fill(theData->getPhi(),theData->getTotalMB());
  hmgr->getHistoProf2(30)->Fill(theData->getEta(),theData->getPhi(),theData->getTotalMB());

}


void MaterialBudgetHistos::endOfRun() 
{
  edm::LogInfo("MaterialBudget") << "MaterialBudgetHistos: Writing Histos ROOT file to" << theFileName;
  hmgr->save( theFileName );
}

