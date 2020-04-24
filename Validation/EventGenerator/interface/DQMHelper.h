#ifndef Validation_EventGenerator_DQMHelper 
#define Validation_EventGenerator_DQMHelper

/* class DQMHelper
 *
 * Simple  class to configure dqm histograms
 *
 */

#include <iostream>
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

class DQMHelper{
 public: 
  DQMHelper(DQMStore::IBooker *i);
  virtual ~DQMHelper();

  MonitorElement* book1dHisto(std::string name,std::string title,int n,double xmin,double xmax,std::string xaxis, std::string yaxis);
  MonitorElement* book2dHisto(std::string name,std::string title,int nx,double xmin,double xmax,int ny,double ymin,double ymax,std::string xaxis, std::string yaxis);
  
  MonitorElement* book1dHisto(const std::string &name,const std::string &title,int n,double xmin,double xmax);
  MonitorElement* book2dHisto(const std::string &name,const std::string &title,int nx,double xmin,double xmax,int ny,double ymin,double ymax);
  
 private:
  DQMStore::IBooker *ibooker;
};

#endif

