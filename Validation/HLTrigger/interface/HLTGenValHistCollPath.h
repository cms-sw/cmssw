#ifndef Validation_HLTrigger_HLTGenValHistCollPath_h
#define Validation_HLTrigger_HLTGenValHistCollPath_h

//********************************************************************************
//
// Description:
//   This contains a collection of HLTGenValHistCollFilter used to measure the efficiencies of all
//   filters in a specified path. The actual booking and filling happens in the respective HLTGenValHistCollFilters.
//
// Author : Finn Labe, UHH, Oct. 2021
// (Heavily borrowed from Sam Harpers HLTDQMFilterEffHists)
//
//***********************************************************************************

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/HLTReco/interface/TriggerEvent.h"

#include "DQMOffline/Trigger/interface/FunctionDefs.h"
#include "DQMOffline/Trigger/interface/UtilFuncs.h"

#include "Validation/HLTrigger/interface/HLTGenValHistCollFilter.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "Validation/HLTrigger/interface/HLTGenValObject.h"

// class containing a collection of HLTGenValHistCollFilters for a specific path
// at object creation time, the object type (used for systematically naming the histogram),
// triggerPath, hltConfig and dR2limit (for deltaR matching) need to be specified
// functions for initial booking of hists, and filling of hists for a single object, are available
class HLTGenValHistCollPath {
public:
  using MonitorElement = dqm::legacy::MonitorElement;
  using DQMStore = dqm::legacy::DQMStore;

  explicit HLTGenValHistCollPath(edm::ParameterSet pathCollConfig, HLTConfigProvider& hltConfig);

  static edm::ParameterSetDescription makePSetDescription();

  void bookHists(DQMStore::IBooker& iBooker,
                 std::vector<edm::ParameterSet>& histConfigs,
                 std::vector<edm::ParameterSet>& histConfigs2D);
  void fillHists(const HLTGenValObject& obj, edm::Handle<trigger::TriggerEvent>& triggerEvent);

private:
  std::string triggerPath_;
  std::vector<HLTGenValHistCollFilter> collectionFilter_;
  std::vector<std::string> filters_;
  HLTConfigProvider hltConfig_;
  bool doOnlyLastFilter_;

  // we will add a string to the root file that is named after the path
  // it will contain the filters of that path divided by semicola
  std::string pathStringName_;
  std::string pathString_;
};

#endif
