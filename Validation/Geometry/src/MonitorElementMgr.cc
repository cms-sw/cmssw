#include "Validation/Geometry/interface/MonitorElementMgr.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#ifdef StatTesting
#include "Validation/SimG4GeometryValidation/interface/StatisticsComparator.h"
#include "StatisticsTesting/Chi2ComparisonAlgorithm.h"
#endif

#ifdef PI121
#include "StatisticsTesting/ComparisonResult.h"
#endif
#include <iostream>
#include "DQMServices/Core/interface/MonitorElement.h"
#include <cstdlib>

//----------------------------------------------------------------------
MonitorElementMgr::MonitorElementMgr()
{

}

//----------------------------------------------------------------------
MonitorElementMgr::~MonitorElementMgr()
{
  //~MonitorElement is protected => delete doesnt work...
  /*
  mime::const_iterator iter;
  for( iter = theMonitorElements.begin(); iter != theMonitorElements.end(); iter++ ){ 
    delete (*iter).second;
  }
  */
}

//----------------------------------------------------------------------
void MonitorElementMgr::save( const std::string& name )
{

  std::cout << "=== save user histos ===" << std::endl;
  TFile fout(name.c_str(),"recreate");
  // write out the histos
mime::const_iterator iter;
 std::cout << "Should save " << theMonitorElements.size() << " MEs" << std::endl;
 /*
   for( iter = theMonitorElements.begin(); iter != theMonitorElements.end(); iter++ ){ 
   delete (*iter).second;
   }
 */
}

void MonitorElementMgr::openSecondFile( const std::string& name )
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

void MonitorElementMgr::printComparisonResult( int ih )
{
  //still to be implemented for MonitorElements
}

bool MonitorElementMgr::addME( MonitorElement* sih )
{
  const char* str = const_cast<char *>((sih->getName()).c_str());
  int ih = atoi(str);
  theMonitorElements[ih] = sih;
  std::cout << " addME " << sih->getName() << " = " << ih << std::endl;

  return true;
}

MonitorElement* MonitorElementMgr::getME( int ime )
{
  MonitorElement* me = 0;
  mime::const_iterator ite = theMonitorElements.find( ime );
  if( ite != theMonitorElements.end() ) {
    me = (*ite).second;
  } else {
    std::cerr << "!!!! FATAL ERROR MonitorElement does not exist " << ime << std::endl;
    std::exception();
  }
  return me;
}

MonitorElement* MonitorElementMgr::getMEFromSecondFile( const char* hnam )
{
  std::cout << "still to be added for ME" << std::endl;
  MonitorElement* me = 0;
  std::cerr << "!!!! FATAL ERROR MonitorElement does not exist " << std::endl;
  std::exception();
  return me;
}
