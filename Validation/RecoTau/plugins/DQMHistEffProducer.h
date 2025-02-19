#ifndef TauDQMHistEffProducer_h
#define TauDQMHistEffProducer_h

/** \class TauDQMHistEffProducer
 *  
 *  Class to produce efficiency histograms by dividing nominator by denominator histograms
 *
 *  $Date: 2012/04/20 13:26:14 $
 *  $Revision: 1.3 $
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

class TauDQMHistEffProducer : public edm::EDAnalyzer
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
  explicit TauDQMHistEffProducer(const edm::ParameterSet&);
  virtual ~TauDQMHistEffProducer();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob(){}
  virtual void endRun(const edm::Run& r, const edm::EventSetup& c);

private:
  std::vector<cfgEntryPlot> cfgEntryPlot_;
  std::vector<MonitorElement*> histoEfficiencyVector_;
};

#endif


