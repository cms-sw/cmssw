#ifndef ElectroWeakAnalysis_EWKTau_DQMSimpleFileSaver_h
#define ElectroWeakAnalysis_EWKTau_DQMSimpleFileSaver_h

/** \class DQMSimpleFileSaver
 *  
 *  Class to write all monitor elements registered in DQMStore into ROOT file
 *  (without any naming restrictions imposed by "regular" DQMFileSaver)
 *
 *  $Date: 2008/12/19 19:05:50 $
 *  $Revision: 1.1 $
 *  \author Christian Veelken, UC Davis
 */

// framework & common header files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>

class DQMSimpleFileSaver : public edm::EDAnalyzer
{
 public:
  explicit DQMSimpleFileSaver(const edm::ParameterSet&);
  virtual ~DQMSimpleFileSaver();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();  

private:
  std::string outputFileName_;
  int cfgError_;
};

#endif


