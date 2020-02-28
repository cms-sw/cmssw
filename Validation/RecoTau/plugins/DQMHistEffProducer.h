#ifndef TauDQMHistEffProducer_h
#define TauDQMHistEffProducer_h

/** \class TauDQMHistEffProducer
 *  
 *  Class to produce efficiency histograms by dividing nominator by denominator histograms
 *
 *  \author Christian Veelken, UC Davis
 */

// framework & common header files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include <string>
#include <vector>

class TauDQMHistEffProducer : public DQMEDHarvester {
  struct cfgEntryPlot {
    explicit cfgEntryPlot(const edm::ParameterSet&);
    explicit cfgEntryPlot(const std::string&, const std::string&, const std::string&);
    std::string numerator_;
    std::string denominator_;
    std::string efficiency_;
  };

public:
  explicit TauDQMHistEffProducer(const edm::ParameterSet&);
  ~TauDQMHistEffProducer() override;
  void dqmEndJob(DQMStore::IBooker& ibook, DQMStore::IGetter& iget) override;

private:
  std::vector<cfgEntryPlot> cfgEntryPlot_;
  std::vector<MonitorElement*> histoEfficiencyVector_;
};

#endif
