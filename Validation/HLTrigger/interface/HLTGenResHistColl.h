#ifndef Validation_HLTrigger_HLTGenValHistCollFilter_h
#define Validation_HLTrigger_HLTGenValHistCollFilter_h

//********************************************************************************
//
// Description:
//   This class contains a collection of HLTGenValHists used to measure the resolution a trigger object type.
//   The trigger object can optionally be required to pass a specific filter.
//
// Author : Sam Harper (RAL), April 2024
//
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
class HLTGenResHistColl {
public:
  using MonitorElement = dqm::legacy::MonitorElement;
  using DQMStore = dqm::legacy::DQMStore;

  explicit HLTGenResHistColl(edm::ParameterSet filterCollConfig, std::string hltProcessName);

  static edm::ParameterSetDescription makePSetDescription();

  void bookHists(DQMStore::IBooker& iBooker);
  void fillHists(const HLTGenValObject& obj, edm::Handle<trigger::TriggerEvent>& triggerEvent);

  const std::string& objType() const { return objType_; }

private:
  void book1D(DQMStore::IBooker& iBooker, const edm::ParameterSet& histConfig);
  void book2D(DQMStore::IBooker& iBooker, const edm::ParameterSet& histConfig2D);
  bool passFilterSelection(trigger::size_type key, const trigger::TriggerEvent& triggerEvent) const;
  std::string getHistName(const std::string& resVar, const std::string& vsVar = "") const;

  std::vector<std::unique_ptr<HLTGenValHist>> hists_;
  std::string objType_;
  bool isEventLevelVariable_;
  std::string tag_;
  std::vector<std::string> filters_;
  bool andFilters_;
  std::string hltProcessName_;
  std::string collectionName_;
  double dR2limit_;
  std::string histNamePrefix_;
  std::string separator_;
  std::vector<edm::ParameterSet> histConfigs_;
  std::vector<edm::ParameterSet> histConfigs2D_;
};

#endif
