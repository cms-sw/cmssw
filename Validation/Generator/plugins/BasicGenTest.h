#ifndef BasicGenTest_h
#define BasicGenTest_h

/*class BasicGenTest
 *  
 *  Class to fill Event Generator dqm monitor elements; works on HepMCProduct
 *
 *  BasicGenTest:
 *  $Date: 2009/07/09 04:23:19 $
 *  $Revision: 1.1 $
 *  \author Joseph Zennamo SUNY-Buffalo; Based on: ConverterTester*/

// framework & common header files
//#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <iostream>
#include <stdlib.h>


class BasicGenTest : public edm::EDAnalyzer
{
  
 public:

  explicit BasicGenTest(const edm::ParameterSet&);
  virtual ~BasicGenTest();
  virtual void beginJob(const edm::EventSetup&);
  virtual void endJob();  
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
  virtual void endRun(const edm::Run&, const edm::EventSetup&);

  
private:
  
  int part_counter[100];

  int glunum, dusnum, cnum, bnum, topnumber, Wnum, Znum; /// charstanum;
  
  DQMStore *dbe;
  
  MonitorElement *gluonNumber;
  MonitorElement *dusNumber;
  MonitorElement *cNumber;
  MonitorElement *bNumber;
  MonitorElement *tNumber;
  MonitorElement *WNumber;
  MonitorElement *ZNumber;
  MonitorElement *stablepart; 
  MonitorElement *Part_ID;
  //  MonitorElement *ChargStableNumber;
};

#endif
