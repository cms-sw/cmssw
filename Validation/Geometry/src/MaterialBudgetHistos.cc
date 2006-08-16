#include "Validation/Geometry/interface/MaterialBudgetHistos.h"
#include "Validation/Geometry/interface/MaterialBudgetData.h"

#include "Validation/Geometry/interface/TestHistoMgr.h"

#include "SimG4Core/Notification/interface/Singleton.h"


MaterialBudgetHistos::MaterialBudgetHistos(MaterialBudgetData* data, const std::string& fileName ): MaterialBudgetFormat( data )
{
  hmgr = Singleton<TestHistoMgr>::instance();
  theFileName = fileName;
  book();

}


void MaterialBudgetHistos::book() 
{
  std::cout << "=== booking user histos ===" << std::endl;
  hmgr->addHistoProf1( new TProfile("10", "MB prof Eta ", 250, -5., 5. ) );
  hmgr->addHisto1( new TH1F("11", "Eta " , 501, -5., 5. ) );
  hmgr->addHistoProf1( new TProfile("20", "MB prof Phi ", 100, -180., 180. ) );
  hmgr->addHisto1( new TH1F("21", "Phi " , 300, -180.01, 179.99 ) );
  hmgr->addHistoProf2( new TProfile2D("30", "MB prof Eta  Phi ", 250, -5., 5., 100, -180., 180. ) );
  hmgr->addHisto2( new TH2F("31", "Eta vs Phi " , 501, -5., 5., 300, -180.01, 179.99 ) );

  std::cout << "=== booking user histos done ===" << std::endl;

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


void MaterialBudgetHistos::hend() 
{
  std::cout << "=== save user histos ===" << std::endl;
  hmgr->save( theFileName );

}

