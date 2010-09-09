//#include "Utilities/Configuration/interface/Architecture.h"
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
  mih1::const_iterator ite1;
  mih2::const_iterator ite2;
  mihp1::const_iterator itep1;
  mihp2::const_iterator itep2;

  for( ite1 = theHistos1.begin(); ite1 != theHistos1.end(); ite1++ ){ 
    delete (*ite1).second;
  }
  for( ite2 = theHistos2.begin(); ite2 != theHistos2.end(); ite2++ ){
    delete (*ite2).second;
  }
  for( itep1 = theHistoProfs1.begin(); itep1 != theHistoProfs1.end(); itep1++ ){
    delete (*itep1).second;
  }
  for( itep2 = theHistoProfs2.begin(); itep2 != theHistoProfs2.end(); itep2++ ){    delete (*itep2).second;
  }
 
}


//----------------------------------------------------------------------
void TestHistoMgr::save( const std::string& name )
{

  std::cout << "=== save user histos ===" << std::endl;
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

  theFileRef = new TFile(name.c_str());

  /*  std::vector<std::string> objectNames = theStoreIn->listObjectNames();
  std::vector<std::string> objectTypes = theStoreIn->listObjectTypes(); 
  unsigned int siz = objectNames.size();
  for( unsigned int ii = 0; ii < siz; ii++ ) {
    //    std::cout << " HISTOS IN FILE " << std::endl;

    //   std::cout << " HISTO: " << objectNames[ii] << " " << objectTypes[ii] << std::endl;
    }*/
}


void TestHistoMgr::printComparisonResult( int ih )
{

  /*
    TH1F* histo1 = getHisto1(ih);
  */
  /*  std::cout << ih << " Histo1 " << histo1;
      std::cout << histo1->GetName();
      std::cout << histo1->title();
      std::cout << histo1->entries();
      std::cout << histo1->axis().bins() << std::endl;
  */
  
#ifdef StatTesting
  TH1F* histo2 = getHisto1FromSecondFile(histo1->GetName());
  
  StatisticsTesting::StatisticsComparator< StatisticsTesting::Chi2ComparisonAlgorithm > comparator;

//  std::cout << " PrintComparisonResult " << histo1 << " " << histo2 << std::endl;
#ifdef PI121
qStatisticsTesting::ComparisonResult result = comparator.compare(*histo1, *histo2); 
#else
  double result = comparator.compare(*histo1, *histo2); 
#endif

  //  std::cout << " PrintComparisonResult " << histo1 << " " << histo2 << std::endl;

  //  double distance  = comparator.calculateDistance(histo1, histo2);
  // ---------------------------------------------------------
  // Do something with (e.g. print out) the result of the test 
  // ---------------------------------------------------------
  std::cout << " Result of the Chi2 Statistical Test: " << histo1->GetName() << std::endl
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
  /*  std::cout << " LOG histo " << histo1->GetName() << " mean = " << histo1->mean() << " rms = " << histo1->rms() << std::endl;
  std::cout << " REF histo " << histo2->GetName() << " mean = " << histo2->mean() << " rms = " << histo2->rms() << std::endl;
  std::cout << " [OVAL]: " << histo1->GetName() << " DIFF_MEAN " << fabs(histo1->mean()-histo2->mean()) << " DIFF_RMS " << fabs(histo1->rms()-histo2->rms()) << std::endl;
  */

#endif
}


bool TestHistoMgr::addHisto1( TH1F* sih )
{
  int ih = atoi(sih->GetName());
  theHistos1[ih] = sih;
  std::cout << " addHisto1 " << sih->GetName() << " = " << ih << std::endl;

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
  std::cout << " addProfHisto1 " << sih->GetName() << " = " << ih << std::endl;

  return true;
}

bool TestHistoMgr::addHistoProf2( TProfile2D* sih )
{
  int ih = atoi(sih->GetName());
  theHistoProfs2[ih] = sih;
  std::cout << " addProfHisto2 " << sih->GetName() << " = " << ih << std::endl;

  return true;
}


TH1F* TestHistoMgr::getHisto1( int ih )
{
  TH1F* his = 0;

  mih1::const_iterator ite = theHistos1.find( ih );
  if( ite != theHistos1.end() ) {
    his = (*ite).second;
  } else {
    std::cerr << "!!!! FATAL ERROR Histogram does not exist " << ih << std::endl;
    std::exception();
  }
  return his;

}

TH2F* TestHistoMgr::getHisto2( int ih )
{
  TH2F* his = 0;
  mih2::const_iterator ite = theHistos2.find( ih );
  if( ite != theHistos2.end() ) {
    his = (*ite).second;
  } else {
    std::cerr << "!!!! FATAL ERROR Histogram does not exist " << ih << std::endl;
    std::exception();
  }
  return his;
}

TProfile* TestHistoMgr::getHistoProf1( int ih )
{
  TProfile* his = 0;
  mihp1::const_iterator ite = theHistoProfs1.find( ih );
  if( ite != theHistoProfs1.end() ) {
   his = (*ite).second;
  } else {
    std::cerr << "!!!! FATAL ERROR Profile Histogram 1D does not exist " << ih << std::endl;
    std::exception();
  }

  return his;

}


TProfile2D* TestHistoMgr::getHistoProf2( int ih )
{
  TProfile2D* his = 0;
  mihp2::const_iterator ite = theHistoProfs2.find( ih );
  if( ite != theHistoProfs2.end() ) {
   his = (*ite).second;
  } else {
    std::cerr << "!!!! FATAL ERROR Profile Histogram 2D does not exist " << ih << std::endl;
    std::exception();
  }

  return his;

}
 
TH1F* TestHistoMgr::getHisto1FromSecondFile( const char* hnam )
{
  if( !theFileRef ){
    std::cerr << "!!!! FATAL ERROR Second file not yet opened " << std::endl;
    std::exception();
  }


  TH1F* his = (TH1F*)(*theFileRef).Get(hnam);
  if( !his ) {
    std::cerr << "!!!! FATAL ERROR Histogram does not exist in second file " << hnam << std::endl;
    theFileRef->ls();
    std::exception();
    /*  } else {
        std::cout << " getHisto1FromSecondFile " << std::endl;
    std::cout << his->GetName() << std::endl;
    std::cout << his->title() << std::endl;
    std::cout << his->entries() << std::endl;
    std::cout << his->axis().bins() << std::endl;
    std::cout << " end getHisto1FromSecondFile " << std::endl;
    */
  }

  return his;

}
