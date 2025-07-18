#include "Validation/HLTrigger/interface/HLTGenResHistColl.h"

namespace {
  // function to get the trigger objects of a specific collection
  std::pair<trigger::size_type, trigger::size_type> getTrigObjIndicesOfCollection(
      const trigger::TriggerEvent& triggerEvent, const std::string& collectionName, const std::string& hltProcessName) {
    trigger::size_type begin = 0;
    trigger::size_type end = 0;
    for (trigger::size_type collNr = 0; collNr < triggerEvent.sizeCollections(); ++collNr) {
      std::string collection = triggerEvent.collectionTag(collNr).label();
      if (collection == collectionName) {
        if (hltProcessName.empty() || triggerEvent.collectionTag(collNr).process() == hltProcessName) {
          if (collNr == 0) {
            begin = 0;
          } else {
            begin = triggerEvent.collectionKey(collNr - 1);
          }
          end = triggerEvent.collectionKey(collNr);
          break;
        }
      }
    }
    return std::make_pair(begin, end);
  }

  bool passesFilter(trigger::size_type objKey,
                    const trigger::TriggerEvent& trigEvent,
                    const std::string& filterName,
                    const std::string& hltProcessName) {
    edm::InputTag filterTag(filterName, "", hltProcessName);
    trigger::size_type filterIndex = trigEvent.filterIndex(filterTag);
    if (filterIndex < trigEvent.sizeFilters()) {  //check that filter is in triggerEvent
      const trigger::Keys& trigKeys = trigEvent.filterKeys(filterIndex);
      return std::find(trigKeys.begin(), trigKeys.end(), objKey) != trigKeys.end();
    } else {
      return false;
    }
  }
  template <typename InType, typename OutType>
  std::vector<OutType> convertVec(const std::vector<InType>& inVec) {
    std::vector<OutType> outVec;
    outVec.reserve(inVec.size());
    for (const auto& inVal : inVec) {
      outVec.push_back(static_cast<OutType>(inVal));
    }
    return outVec;
  }
}  // namespace

// constructor
HLTGenResHistColl::HLTGenResHistColl(edm::ParameterSet filterCollConfig, std::string hltProcessName)
    : hltProcessName_(hltProcessName) {
  objType_ = filterCollConfig.getParameter<std::string>("objType");
  isEventLevelVariable_ = objType_ == "MET" || objType_ == "AK4HT" || objType_ == "AK8HT";
  filters_ = filterCollConfig.getParameter<std::vector<std::string>>("filterNames");
  andFilters_ = filterCollConfig.getParameter<bool>("andFilters");
  collectionName_ = filterCollConfig.getParameter<std::string>("collectionName");
  dR2limit_ = filterCollConfig.getParameter<double>("dR2limit");
  histConfigs_ = filterCollConfig.getParameter<std::vector<edm::ParameterSet>>("histConfigs");
  histConfigs2D_ = filterCollConfig.getParameter<std::vector<edm::ParameterSet>>("histConfigs2D");
  histNamePrefix_ = "resHist";
  separator_ = "__";
}

edm::ParameterSetDescription HLTGenResHistColl::makePSetDescription() {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("objType", "");
  desc.add<std::vector<std::string>>("filterNames", {});
  desc.add<bool>("andFilters", true);
  desc.add<std::string>("collectionName", "");
  desc.add<double>("dR2limit", 0.1);
  std::vector<edm::ParameterSet> histConfigDefaults;

  edm::ParameterSetDescription histConfig;
  histConfig.add<std::string>("vsVar");
  histConfig.add<std::string>("resVar");
  histConfig.add<std::vector<double>>("vsBinLowEdges");
  histConfig.add<std::vector<double>>("resBinLowEdges");
  histConfig.addVPSet(
      "rangeCuts", VarRangeCut<HLTGenValObject>::makePSetDescription(), std::vector<edm::ParameterSet>());

  edm::ParameterSet histConfigDefault0;
  histConfigDefault0.addParameter<std::string>("resVar", "ptRes");
  histConfigDefault0.addParameter<std::string>("vsVar", "pt");
  std::vector<double> defaultPtBinning{0,  5,  10, 12.5, 15,  17.5, 20,  22.5, 25,  30,  35, 40,
                                       45, 50, 60, 80,   100, 150,  200, 250,  300, 350, 400};
  histConfigDefault0.addParameter<std::vector<double>>("vsBinLowEdges", defaultPtBinning);
  std::vector<double> defaultResBinning;
  defaultResBinning.reserve(151);
  for (int i = 0; i < 151; i++) {
    defaultResBinning.push_back(i * 0.01);
  }
  histConfigDefault0.addParameter<std::vector<double>>("resBinLowEdges", defaultResBinning);

  histConfigDefaults.push_back(histConfigDefault0);
  desc.addVPSet("histConfigs", histConfig, histConfigDefaults);

  // defining single histConfig2D
  edm::ParameterSetDescription histConfig2D;
  histConfig2D.add<std::string>("vsVarX");
  histConfig2D.add<std::string>("vsVarY");
  histConfig2D.add<std::vector<double>>("binLowEdgesX");
  histConfig2D.add<std::vector<double>>("binLowEdgesY");
  histConfig2D.addVPSet(
      "rangeCuts", VarRangeCut<HLTGenValObject>::makePSetDescription(), std::vector<edm::ParameterSet>());
  // default set of histConfigs
  std::vector<edm::ParameterSet> histConfigDefaults2D;
  desc.addVPSet("histConfigs2D", histConfig2D, histConfigDefaults2D);

  return desc;
}

bool HLTGenResHistColl::passFilterSelection(trigger::size_type objKey,
                                            const trigger::TriggerEvent& triggerEvent) const {
  if (andFilters_) {
    for (const auto& filter : filters_) {
      if (!passesFilter(objKey, triggerEvent, filter, hltProcessName_)) {
        return false;
      }
    }
    return true;
  } else {
    for (const auto& filter : filters_) {
      if (passesFilter(objKey, triggerEvent, filter, hltProcessName_)) {
        return true;
      }
    }
    return false;
  }
}

// general hist booking function, receiving configurations for 1D and 2D hists and calling the respective functions
void HLTGenResHistColl::bookHists(DQMStore::IBooker& iBooker) {
  for (const auto& histConfig : histConfigs_)
    book1D(iBooker, histConfig);
  for (const auto& histConfig : histConfigs2D_)
    book2D(iBooker, histConfig);
}

// histogram filling routine
void HLTGenResHistColl::fillHists(const HLTGenValObject& obj, edm::Handle<trigger::TriggerEvent>& triggerEvent) {
  // get the trigger objects of the collection
  auto keyRange = getTrigObjIndicesOfCollection(*triggerEvent, collectionName_, hltProcessName_);
  const trigger::TriggerObject* bestMatch = nullptr;
  float bestDR2 = dR2limit_;
  for (trigger::size_type key = keyRange.first; key < keyRange.second; ++key) {
    if (passFilterSelection(key, *triggerEvent)) {
      const trigger::TriggerObject objTrig = triggerEvent->getObjects().at(key);
      if (isEventLevelVariable_) {
        bestDR2 = 0;
        bestMatch = &objTrig;
        break;
      } else {
        float dR2 = reco::deltaR2(obj, objTrig);
        if (dR2 < bestDR2) {
          bestDR2 = dR2;
          bestMatch = &objTrig;
        }
      }
    }
  }
  HLTGenValObject objWithTrig(obj);
  if (bestMatch) {
    objWithTrig.setTrigObject(*bestMatch);
    for (auto& hist : hists_)
      hist->fill(objWithTrig);
  }
}

// booker function for 1D hists
void HLTGenResHistColl::book1D(DQMStore::IBooker& iBooker, const edm::ParameterSet& histConfig) {
  // extracting parameters from configuration
  auto vsVar = histConfig.getParameter<std::string>("vsVar");
  auto vsVarFunc = hltdqm::getUnaryFuncFloat<HLTGenValObject>(vsVar);
  auto resVar = histConfig.getParameter<std::string>("resVar");
  auto resVarFunc = hltdqm::getUnaryFuncFloat<HLTGenValObject>(resVar);

  // getting the bin edges, path-specific overwrites general if present
  auto resBinLowEdgesDouble = histConfig.getParameter<std::vector<double>>("resBinLowEdges");
  auto vsBinLowEdgesDouble = histConfig.getParameter<std::vector<double>>("vsBinLowEdges");

  // additional cuts applied to this histogram, combination of general ones and path-specific ones
  std::vector<edm::ParameterSet> allCutsVector = histConfig.getParameter<std::vector<edm::ParameterSet>>("rangeCuts");
  VarRangeCutColl<HLTGenValObject> rangeCuts(allCutsVector);

  // getting the custom tag
  const std::string tag;

  // checking validity of vsVar
  if (!vsVarFunc) {
    throw cms::Exception("ConfigError") << " vsVar " << vsVar << " is giving null ptr (likely empty) in " << __FILE__
                                        << "," << __LINE__ << std::endl;
  }

  // converting bin edges to float
  std::vector<float> vsBinLowEdges = convertVec<double, float>(vsBinLowEdgesDouble);
  std::vector<float> resBinLowEdges = convertVec<double, float>(resBinLowEdgesDouble);

  std::string histNameInc = getHistName(resVar);
  std::string histTitleInc = collectionName_ + "to " + objType_ + " " + resVar;
  if (!tag.empty()) {
    histNameInc += separator_ + tag;
    histTitleInc += " " + tag;
  }
  std::string histNameVsBase = getHistName(resVar, vsVar);
  std::string histName2D = histNameVsBase + separator_ + "2D";
  std::string histNameProfile = histNameVsBase + separator_ + "profile";

  std::string histTitle2D = collectionName_ + "to " + objType_ + " " + resVar + " vs " + vsVar;
  std::string histTitleProfile = histTitle2D;
  if (!tag.empty()) {
    histName2D += separator_ + tag;
    histTitle2D += " " + tag;
    histNameProfile += separator_ + tag;
    histTitleProfile += " " + tag;
  }

  auto resInc =
      iBooker.book1D(histNameInc.c_str(), histTitleInc.c_str(), resBinLowEdges.size() - 1, resBinLowEdges.data());

  auto res2D = iBooker.book2D(histName2D.c_str(),
                              histTitle2D.c_str(),
                              vsBinLowEdges.size() - 1,
                              vsBinLowEdges.data(),
                              resBinLowEdges.size() - 1,
                              resBinLowEdges.data());

  //so bookProfile requires double bin edges, not float like book2D...
  auto resProf = iBooker.bookProfile(histNameProfile.c_str(),
                                     histTitleProfile.c_str(),
                                     vsBinLowEdgesDouble.size() - 1,
                                     vsBinLowEdgesDouble.data(),
                                     0.2,
                                     5);

  std::unique_ptr<HLTGenValHist> hist;

  hist = std::make_unique<HLTGenValHist2D>(
      res2D->getTH2F(), resProf->getTProfile(), vsVar, resVar, vsVarFunc, resVarFunc, rangeCuts);
  hists_.emplace_back(std::move(hist));
  hist = std::make_unique<HLTGenValHist1D>(resInc->getTH1(), resVar, resVarFunc, rangeCuts);
  hists_.emplace_back(std::move(hist));
}

// booker function for 2D hists
void HLTGenResHistColl::book2D(DQMStore::IBooker& iBooker, const edm::ParameterSet& histConfig2D) {
  // extracting parameters from configuration
  auto vsVarX = histConfig2D.getParameter<std::string>("vsVarX");
  auto vsVarY = histConfig2D.getParameter<std::string>("vsVarY");
  auto vsVarFuncX = hltdqm::getUnaryFuncFloat<HLTGenValObject>(vsVarX);
  auto vsVarFuncY = hltdqm::getUnaryFuncFloat<HLTGenValObject>(vsVarY);
  auto binLowEdgesDoubleX = histConfig2D.getParameter<std::vector<double>>("binLowEdgesX");
  auto binLowEdgesDoubleY = histConfig2D.getParameter<std::vector<double>>("binLowEdgesY");

  // checking validity of vsVar
  if (!vsVarFuncX) {
    throw cms::Exception("ConfigError") << " vsVar " << vsVarX << " is giving null ptr (likely empty) in " << __FILE__
                                        << "," << __LINE__ << std::endl;
  }
  if (!vsVarFuncY) {
    throw cms::Exception("ConfigError") << " vsVar " << vsVarY << " is giving null ptr (likely empty) in " << __FILE__
                                        << "," << __LINE__ << std::endl;
  }

  // converting bin edges to float
  std::vector<float> binLowEdgesX;
  std::vector<float> binLowEdgesY;
  binLowEdgesX.reserve(binLowEdgesDoubleX.size());
  binLowEdgesY.reserve(binLowEdgesDoubleY.size());
  for (double lowEdge : binLowEdgesDoubleX)
    binLowEdgesX.push_back(lowEdge);
  for (double lowEdge : binLowEdgesDoubleY)
    binLowEdgesY.push_back(lowEdge);

  // name and t
  std::string histName =
      objType_ + separator_ + separator_ + "GEN" + separator_ + "2Dvs" + vsVarX + separator_ + vsVarY;
  std::string histTitle = objType_ + " GEN 2D vs " + vsVarX + " " + vsVarY;

  auto me = iBooker.book2D(histName.c_str(),
                           histTitle.c_str(),
                           binLowEdgesX.size() - 1,
                           binLowEdgesX.data(),
                           binLowEdgesY.size() - 1,
                           binLowEdgesY.data());

  std::unique_ptr<HLTGenValHist> hist;

  hist = std::make_unique<HLTGenValHist2D>(me->getTH2F(), vsVarX, vsVarY, vsVarFuncX, vsVarFuncY);

  hists_.emplace_back(std::move(hist));
}

std::string HLTGenResHistColl::getHistName(const std::string& resVar, const std::string& vsVar) const {
  std::string histName = histNamePrefix_ + separator_ + collectionName_ + separator_ + objType_ + separator_ + resVar;
  if (!vsVar.empty()) {
    histName += separator_ + "vs" + separator_ + vsVar;
  }
  return histName;
}
