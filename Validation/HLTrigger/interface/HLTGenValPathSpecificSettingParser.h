#ifndef Validation_HLTrigger_HLTGenValPathSpecificSettingParser_h
#define Validation_HLTrigger_HLTGenValPathSpecificSettingParser_h

//********************************************************************************
//
// Description:
//   This class handles parsing of additional settings that can be set for each path in the generator-level validation module
//   Mainly, these are cuts in addition to the ones specified in the module. Passing a pre-defined region is also possible
//   The binning of a certain variable can be changed through this class, as well as setting a tag for all histograms of a path.
//
// Author : Finn Labe, UHH, Jul. 2022
//
//********************************************************************************

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include <vector>

class HLTGenValPathSpecificSettingParser {
public:
  // constructor
  HLTGenValPathSpecificSettingParser(std::string pathSpecificSettings,
                                     std::vector<edm::ParameterSet> binnings,
                                     std::string vsVar);

  const std::vector<edm::ParameterSet>* getPathSpecificCuts() const { return &pathSpecificCutsVector_; }
  const std::vector<double>* getPathSpecificBins() const { return &pathSpecificBins_; }
  const bool havePathSpecificBins() const { return (!pathSpecificBins_.empty()); }
  const std::string* getTag() const { return &tag_; }

private:
  std::vector<edm::ParameterSet> pathSpecificCutsVector_;
  std::vector<double> pathSpecificBins_;
  std::string tag_ = "";
};

#endif
