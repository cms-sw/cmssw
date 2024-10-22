#ifndef Validation_HLTrigger_HLTGenValHistCollFilter_h
#define Validation_HLTrigger_HLTGenValHistCollFilter_h

//********************************************************************************
//
// Description:
//   This class contains a collection of HLTGenvalHists used to measure the efficiency of a
//   specified filter. It is resonsible for booking and filling the histograms of all vsVars
//   histograms that are created for a specific filter.
//
// Author : Finn Labe, UHH, Jul. 2022
//          (Strongly inspired by Sam Harpers HLTDQMFilterEffHists class)
//
//***********************************************************************************

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/HLTReco/interface/TriggerEvent.h"

#include "Validation/HLTrigger/interface/HLTGenValHist.h"
#include "DQMOffline/Trigger/interface/FunctionDefs.h"
#include "DQMOffline/Trigger/interface/UtilFuncs.h"

#include "Validation/HLTrigger/interface/HLTGenValObject.h"
#include "Validation/HLTrigger/interface/HLTGenValPathSpecificSettingParser.h"

#include "DataFormats/Math/interface/deltaR.h"

#include <utility>

// class containing a collection of HLTGenValHist for a specific filter
// functions for initial booking of hists, and filling of hists for a single object are available
class HLTGenValHistCollFilter {
public:
  using MonitorElement = dqm::legacy::MonitorElement;
  using DQMStore = dqm::legacy::DQMStore;

  explicit HLTGenValHistCollFilter(edm::ParameterSet filterCollConfig);

  static edm::ParameterSetDescription makePSetDescription();

  void bookHists(DQMStore::IBooker& iBooker,
                 const std::vector<edm::ParameterSet>& histConfigs,
                 const std::vector<edm::ParameterSet>& histConfigs2D);
  void fillHists(const HLTGenValObject& obj, edm::Handle<trigger::TriggerEvent>& triggerEvent);

private:
  void book1D(DQMStore::IBooker& iBooker, const edm::ParameterSet& histConfig);
  void book2D(DQMStore::IBooker& iBooker, const edm::ParameterSet& histConfig2D);

  std::vector<std::unique_ptr<HLTGenValHist>> hists_;  // the collection of histograms
  std::string objType_;
  std::string tag_;
  std::string filter_;
  std::string path_;
  std::string hltProcessName_;
  double dR2limit_;
  std::string separator_;
};

#endif
