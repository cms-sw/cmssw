#ifndef BasicGenTest_h
#define BasicGenTest_h

/** \class BasicGenTest (followed closely from ConverterTester)
 *  
 *  Class to fill dqm monitor elements from existing EDM file (Mike)
 *
 *  BasicGenTest:
 *  $Date: 2009/07/08 18:35:12 $
 *  $Revision: 0.0-0.1ish $
 *  \author Joseph Zennamo SUNY-Buffalo
 *
 *  ConverterTester:
 *  $Date: 2008/04/16 21:52:41 $
 *  $Revision: 1.2 $
 *  \author M. Strang SUNY-Buffalo
 */

// framework & common header files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <iostream>
#include <stdlib.h>
#include <string>
#include <memory>
#include <vector>

#include "TString.h"
#include "TRandom.h"
#include "TRandom3.h"

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

  int bnum, topnumber, wnum;
  int dusnum, cnum, Znum, charstanum;
  int partc;
  
  DQMStore *dbe;

  MonitorElement *bNumber;
  MonitorElement *meTestInt;
  MonitorElement *particle_number;
  MonitorElement *WNumber;
  MonitorElement *dusNumber;
  MonitorElement *cNumber;
  MonitorElement *tNumber;
  MonitorElement *ZNumber;
  MonitorElement *ChargStableNumber;
  MonitorElement *PartonNumber;
 
 
};

#endif
