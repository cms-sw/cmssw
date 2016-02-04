#ifndef MonitorElementMgr_h
#define MonitorElementMgr_h

#include "DQMServices/Core/interface/MonitorElement.h"
#include "TFile.h"
#include <map>
#include <string>

typedef std::map< int, MonitorElement* > mime;

class MonitorElementMgr
{
 public:
  MonitorElementMgr();
  ~MonitorElementMgr();
  void save( const std::string& name );
  void openSecondFile( const std::string& name );

  void printComparisonResult( int ime );
  
  bool addME( MonitorElement* ime );
  
  MonitorElement* getME( int ime );

  MonitorElement* getMEFromSecondFile( const char* hnam );

 private:
  mime theMonitorElements;

  TFile * theFileRef;
  
};

#endif
