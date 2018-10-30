//#define StatTesting
//#define PI121

#include "Validation/Geometry/interface/TestHistoMgr.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#ifdef StatTesting
#include "Validation/SimG4GeometryValidation/interface/StatisticsComparator.h"
#include "StatisticsTesting/Chi2ComparisonAlgorithm.h"
#endif

#ifdef PI121
#include "StatisticsTesting/ComparisonResult.h"
#endif
#include <iostream>
#include <cstdlib>

//----------------------------------------------------------------------
TestHistoMgr::TestHistoMgr()
{
  //  pi_aida::Proxy_Selector::setHistogramType(pi_aida::Histogram_Native);
}

//----------------------------------------------------------------------
TestHistoMgr::~TestHistoMgr()
{ 
  for(auto& it: theHistos1){
    delete it.second;
  }
  for(auto& it: theHistos2){
    delete it.second;
  }
  for(auto& it: theHistoProfs1){
    delete it.second;
  }
  for(auto& it: theHistoProfs2){
    delete it.second;
  } 
}

//----------------------------------------------------------------------
void TestHistoMgr::save( const std::string& name )
{

  edm::LogInfo("MaterialBudget") << "TestHistoMgr: Save user histos";

  TFile fout(name.c_str(),"recreate");

  // write out the histos
  mih1::const_iterator ite1;
  mih2::const_iterator ite2;
  mihp1::const_iterator itep1;
  mihp2::const_iterator itep2;

  for( ite1 = theHistos1.begin(); ite1 != theHistos1.end(); ite1++ ){
    ((*ite1).second)->Write();
  }
  for( ite2 = theHistos2.begin(); ite2 != theHistos2.end(); ite2++ ){
    ((*ite2).second)->Write();
  }
  for( itep1 = theHistoProfs1.begin(); itep1 != theHistoProfs1.end(); itep1++ ){
    ((*itep1).second)->Write();
  }
  for( itep2 = theHistoProfs2.begin(); itep2 != theHistoProfs2.end(); itep2++ ){
    ((*itep2).second)->Write();
  }

}

void TestHistoMgr::openSecondFile( const std::string& name )
{

  theFileRef = std::make_unique<TFile>(name.c_str());

}


void TestHistoMgr::printComparisonResult( int ih )
{
  
#ifdef StatTesting

  TH1F* histo2 = getHisto1FromSecondFile(histo1->GetName());
  
  StatisticsTesting::StatisticsComparator< StatisticsTesting::Chi2ComparisonAlgorithm > comparator;

#ifdef PI121

  qStatisticsTesting::ComparisonResult result = comparator.compare(*histo1, *histo2); 

#else

  double result = comparator.compare(*histo1, *histo2); 

#endif

  // ---------------------------------------------------------
  // Do something with (e.g. print out) the result of the test 
  // ---------------------------------------------------------
  edm::LogInfo << "TestHistoMgr: Result of the Chi2 Statistical Test: " << histo1->GetName();
#ifdef PI121
  << " distance = " << result.distance() << std::endl
  << " ndf = " << result.ndf() << std::endl
  << " p-value = " << result.quality() << std::endl
#else
            << " p-value = " << result << std::endl
 #endif
            << "---------------- exampleReadXML  ENDS   ------------------ " 
	    << std::endl;
#ifdef PI121
  std::cout << "[OVAL]: HISTO_QUALITY " << histo1->GetName() <<" " << result.quality() << std::endl;
 #else 
  std::cout << "[OVAL]: HISTO_QUALITY " << histo1->GetName() <<" " << result << std::endl;
 #endif
  std::cout << std::endl << " mean= " << histo1->GetMean() << " " << histo1->GetName() << " " << histo1->GetTitle() << std::endl;
  std::cout << " rms= " << histo1->GetRMS() << " " << histo1->GetName() << " " << histo1->GetTitle() << std::endl;

#else
#endif
}


bool TestHistoMgr::addHisto1( TH1F* sih )
{
  int ih = atoi(sih->GetName());
  theHistos1[ih] = sih;
  edm::LogInfo("MaterialBudget") << "TestHistoMgr: addHisto1: " << ih << " = " << sih->GetTitle();
  return true;
}

bool TestHistoMgr::addHisto2( TH2F* sih )
{
  int ih = atoi(sih->GetName());
  theHistos2[ih] = sih;
  return true;
}


bool TestHistoMgr::addHistoProf1( TProfile* sih )
{
  int ih = atoi(sih->GetName());
  theHistoProfs1[ih] = sih;
  edm::LogInfo("MaterialBudget") << "TestHistoMgr: addHistoProf1: " << ih << " = " << sih->GetTitle();
  return true;
}

bool TestHistoMgr::addHistoProf2( TProfile2D* sih )
{
  int ih = atoi(sih->GetName());
  theHistoProfs2[ih] = sih;
  edm::LogInfo("MaterialBudget") << "TestHistoMgr: addHistoProf2: " << ih << " = " << sih->GetTitle();
  return true;
}


TH1F* TestHistoMgr::getHisto1( int ih )
{
  TH1F* his = nullptr;

  mih1::const_iterator ite = theHistos1.find( ih );
  if( ite != theHistos1.end() ) {
    his = (*ite).second;
  } else {
    edm::LogError("MaterialBudget") << "TestHistoMgr: getHisto1 Histogram does not exist " << ih;
    std::exception();
  }
  return his;
}

TH2F* TestHistoMgr::getHisto2( int ih )
{
  TH2F* his = nullptr;
  mih2::const_iterator ite = theHistos2.find( ih );
  if( ite != theHistos2.end() ) {
    his = (*ite).second;
  } else {
    edm::LogError("MaterialBudget") << "TestHistoMgr: getHisto2 Histogram does not exist " << ih;
    std::exception();
  }
  return his;
}

TProfile* TestHistoMgr::getHistoProf1( int ih )
{
  TProfile* his = nullptr;
  mihp1::const_iterator ite = theHistoProfs1.find( ih );
  if( ite != theHistoProfs1.end() ) {
    his = (*ite).second;
  } else {
    edm::LogError("MaterialBudget") << "TestHistoMgr: Profile Histogram 1D does not exist " << ih;
    std::exception();
  }
  return his;
}


TProfile2D* TestHistoMgr::getHistoProf2( int ih )
{
  TProfile2D* his = nullptr;
  mihp2::const_iterator ite = theHistoProfs2.find( ih );
  if( ite != theHistoProfs2.end() ) {
    his = (*ite).second;
  } else {
    edm::LogError("MaterialBudget") << "TestHistoMgr: Profile Histogram 2D does not exist " << ih;
    std::exception();
  }
  return his;
}
 
TH1F* TestHistoMgr::getHisto1FromSecondFile( const char* hnam )
{
  TH1F* his = new TH1F();
  if( !theFileRef ){
    edm::LogError("MaterialBudget") << "TestHistoMgr: Second file not yet opened ";
    std::exception();
  } else{
    his = (TH1F*)(*theFileRef).Get(hnam);
  }

  if( !his ) {
    edm::LogError("MaterialBudget") << "TestHistoMgr: FATAL ERROR Histogram does not exist in second file " << hnam;
    theFileRef->ls();
    std::exception();
  }
  return his;
}
