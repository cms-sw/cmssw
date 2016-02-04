#ifndef ElectroWeakAnalysis_EWKTau_DQMHistEffProducer_h
#define ElectroWeakAnalysis_EWKTau_DQMHistEffProducer_h

/** \class DQMHistEffProducer
 *  
 *  Class to produce efficiency histograms by dividing nominator by denominator histograms
 *
 *  $Date: 2011/04/06 12:20:33 $
 *  $Revision: 1.2 $
 *  \author Christian Veelken, UC Davis
 */

// framework & common header files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include <string>
#include <vector>

class DQMHistEffProducer : public edm::EDAnalyzer
{
  struct cfgEntryPlot
  {
    explicit cfgEntryPlot(const edm::ParameterSet&);
    explicit cfgEntryPlot(const std::string&, const std::string&, const std::string&);
    std::string numerator_;
    std::string denominator_;
    std::string efficiency_;
  };

 public:
  explicit DQMHistEffProducer(const edm::ParameterSet&);
  virtual ~DQMHistEffProducer();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob(){}
  virtual void endRun(const edm::Run& r, const edm::EventSetup& c);

private:
  std::vector<cfgEntryPlot> cfgEntryPlot_;
  std::vector<MonitorElement*> histoEfficiencyVector_;
};

#endif


