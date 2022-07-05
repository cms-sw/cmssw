#include "Validation/HLTrigger/interface/HLTGenValPathSpecificSettingParser.h"

HLTGenValPathSpecificSettingParser::HLTGenValPathSpecificSettingParser(std::string pathSpecificSettings,
                                                                       std::vector<edm::ParameterSet> binnings,
                                                                       std::string vsVar) {
  // splitting the cutstring
  std::stringstream pathSpecificSettingsStream(pathSpecificSettings);
  std::string pathSpecificSettingsSegment;
  std::vector<std::string> pathSpecificSettingsSeglist;
  while (std::getline(pathSpecificSettingsStream, pathSpecificSettingsSegment, ',')) {
    pathSpecificSettingsSeglist.push_back(pathSpecificSettingsSegment);
  }

  for (const auto& pathSpecificSetting : pathSpecificSettingsSeglist) {
    // each of these strings is expected to contain exactly one equal sign
    std::stringstream pathSpecificSettingStream(pathSpecificSetting);
    std::string pathSpecificSettingSegment;
    std::vector<std::string> pathSpecificSettingSeglist;
    while (std::getline(pathSpecificSettingStream, pathSpecificSettingSegment, '=')) {
      pathSpecificSettingSeglist.push_back(pathSpecificSettingSegment);
    }
    if (pathSpecificSettingSeglist.size() != 2)
      throw cms::Exception("InputError") << "Path-specific cuts could not be parsed. Make sure that each parameter "
                                            "contains exactly one equal sign!.\n";
    const std::string cutVariable = pathSpecificSettingSeglist.at(0);
    const std::string cutParameter = pathSpecificSettingSeglist.at(1);

    edm::ParameterSet rangeCutConfig;
    if (cutVariable == "absEtaMax" || cutVariable == "absEtaCut") {
      rangeCutConfig.addParameter<std::string>("rangeVar", "eta");
      rangeCutConfig.addParameter<std::vector<std::string>>("allowedRanges", {"-" + cutParameter + ":" + cutParameter});
    } else if (cutVariable == "absEtaMin") {
      rangeCutConfig.addParameter<std::string>("rangeVar", "eta");
      rangeCutConfig.addParameter<std::vector<std::string>>("allowedRanges",
                                                            {"-999:" + cutParameter, cutParameter + ":999"});
    } else if (cutVariable == "ptMax") {
      rangeCutConfig.addParameter<std::string>("rangeVar", "pt");
      rangeCutConfig.addParameter<std::vector<std::string>>("allowedRanges", {"0:" + cutParameter});
    } else if (cutVariable == "ptMin" || cutVariable == "ptCut") {
      rangeCutConfig.addParameter<std::string>("rangeVar", "pt");
      rangeCutConfig.addParameter<std::vector<std::string>>("allowedRanges", {cutParameter + ":999999"});
    } else if (cutVariable == "etMax") {
      rangeCutConfig.addParameter<std::string>("rangeVar", "et");
      rangeCutConfig.addParameter<std::vector<std::string>>("allowedRanges", {"0:" + cutParameter});
    } else if (cutVariable == "etMin" || cutVariable == "etCut") {
      rangeCutConfig.addParameter<std::string>("rangeVar", "et");
      rangeCutConfig.addParameter<std::vector<std::string>>("allowedRanges", {cutParameter + ":999999"});
    } else if (cutVariable == "region") {
      rangeCutConfig.addParameter<std::string>("rangeVar", "eta");

      // various predefined regions
      // multiple regions might used, which are then split by a plus sign
      std::stringstream cutParameterStream(cutParameter);
      std::string cutParameterSegment;
      std::vector<std::string> cutParameterSeglist;
      while (std::getline(cutParameterStream, cutParameterSegment, '+')) {
        cutParameterSeglist.push_back(cutParameterSegment);
      }

      for (const auto& region : cutParameterSeglist) {
        if (region == "EB") {
          rangeCutConfig.addParameter<std::vector<std::string>>("allowedRanges", {"-1.4442:1.4442"});
        } else if (region == "EE") {
          rangeCutConfig.addParameter<std::vector<std::string>>("allowedRanges", {"-999:-1.5660", "1.5660:999"});
        } else {
          throw cms::Exception("InputError") << "Region " + region + " not recognized.\n";
        }
      }

    } else if (cutVariable == "bins") {
      // sets of binnings are read from the user-input ones passed in the "binnings" VPset

      bool binningFound = false;
      bool binningUsed = false;
      for (const auto& binning : binnings) {
        if (binning.getParameter<std::string>("name") == cutParameter) {
          if (binning.getParameter<std::string>("vsVar") == vsVar) {
            if (binningUsed)
              throw cms::Exception("InputError")
                  << "Multiple different binnings set for a path, this does not make sense!.\n";
            pathSpecificBins_ = binning.getParameter<std::vector<double>>("binLowEdges");
            binningUsed = true;
          }
          binningFound = true;
        }
      }
      if (!binningFound)
        throw cms::Exception("InputError")
            << "Binning " << cutParameter << " not recognized! Please pass the definition to the module.\n";

    } else if (cutVariable == "tag") {
      tag_ = cutParameter;
    } else if (cutVariable == "autotag") {
      // autotag is only used if no manual tag is set
      if (tag_.empty())
        tag_ = cutParameter;
    } else {
      throw cms::Exception("InputError")
          << "Path-specific cut " + cutVariable +
                 " not recognized. The following options can be user: absEtaMax, absEtaCut, absEtaMin, ptMax, ptMin, "
                 "ptCut, etMax, etMin, etCut, region, bins and tag.\n";
    }

    if (!rangeCutConfig.empty())
      pathSpecificCutsVector_.push_back(rangeCutConfig);
  }
}
