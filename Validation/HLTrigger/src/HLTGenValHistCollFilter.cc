#include "Validation/HLTrigger/interface/HLTGenValHistCollFilter.h"

// constructor
HLTGenValHistCollFilter::HLTGenValHistCollFilter(edm::ParameterSet filterCollConfig) {
  objType_ = filterCollConfig.getParameter<std::string>("objType");
  filter_ = filterCollConfig.getParameter<std::string>("filterName");
  path_ = filterCollConfig.getParameter<std::string>("pathName");
  hltProcessName_ = filterCollConfig.getParameter<std::string>("hltProcessName");
  dR2limit_ = filterCollConfig.getParameter<double>("dR2limit");
  separator_ = "__";
}

edm::ParameterSetDescription HLTGenValHistCollFilter::makePSetDescription() {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("objType", "");
  desc.add<std::string>("hltProcessName", "HLT");
  desc.add<double>("dR2limit", 0.1);
  desc.add<std::string>("filterName", "");
  desc.add<std::string>("pathName", "");
  return desc;
}

// general hist booking function, receiving configurations for 1D and 2D hists and calling the respective functions
void HLTGenValHistCollFilter::bookHists(DQMStore::IBooker& iBooker,
                                        const std::vector<edm::ParameterSet>& histConfigs,
                                        const std::vector<edm::ParameterSet>& histConfigs2D) {
  for (const auto& histConfig : histConfigs)
    book1D(iBooker, histConfig);
  for (const auto& histConfig : histConfigs2D)
    book2D(iBooker, histConfig);
}

// histogram filling routine
void HLTGenValHistCollFilter::fillHists(const HLTGenValObject& obj, edm::Handle<trigger::TriggerEvent>& triggerEvent) {
  // this handles the "before" step, denoted by a "dummy" filter called "beforeAnyFilter"
  // the histogram is filled without any additional requirements for all objects
  if (filter_ == "beforeAnyFilter") {
    for (auto& hist : hists_)
      hist->fill(obj);
  } else {
    // main filling code

    // get filter object from filter name
    edm::InputTag filterTag(filter_, "", hltProcessName_);
    size_t filterIndex = triggerEvent->filterIndex(filterTag);

    // get trigger objects passing filter in question
    trigger::TriggerObjectCollection allTriggerObjects = triggerEvent->getObjects();  // all objects
    trigger::TriggerObjectCollection selectedObjects;  // vector to fill with objects passing our filter
    if (filterIndex < triggerEvent->sizeFilters()) {
      const auto& keys = triggerEvent->filterKeys(filterIndex);
      for (unsigned short key : keys) {
        trigger::TriggerObject foundObject = allTriggerObjects[key];
        selectedObjects.push_back(foundObject);
      }
    }

    // differentiate between event level and particle level variables
    const static std::vector<std::string> eventLevelVariables = {"AK4HT", "AK8HT", "MET"};
    if (std::find(eventLevelVariables.begin(), eventLevelVariables.end(), objType_) != eventLevelVariables.end()) {
      // for these event level variables we only require the existence of a trigger object, but no matching
      if (!selectedObjects.empty())
        for (auto& hist : hists_)
          hist->fill(obj);
    } else {
      // do a deltaR matching between trigger object and GEN object
      double mindR2 = 99999;
      for (const auto& filterobj : selectedObjects) {
        double dR = deltaR2(obj, filterobj);
        if (dR < mindR2)
          mindR2 = dR;
      }
      if (mindR2 < dR2limit_)
        for (auto& hist : hists_)
          hist->fill(obj);
    }
  }
}

// booker function for 1D hists
void HLTGenValHistCollFilter::book1D(DQMStore::IBooker& iBooker, const edm::ParameterSet& histConfig) {
  // extracting parameters from configuration
  auto vsVar = histConfig.getParameter<std::string>("vsVar");
  auto vsVarFunc = hltdqm::getUnaryFuncFloat<HLTGenValObject>(vsVar);

  // this class parses any potential additional cuts, changes in binning or tags
  HLTGenValPathSpecificSettingParser parser =
      HLTGenValPathSpecificSettingParser(histConfig.getParameter<std::string>("pathSpecificCuts"),
                                         histConfig.getParameter<std::vector<edm::ParameterSet>>("binnings"),
                                         vsVar);

  // getting the bin edges, path-specific overwrites general if present
  auto binLowEdgesDouble = histConfig.getParameter<std::vector<double>>("binLowEdges");
  if (parser.havePathSpecificBins())
    binLowEdgesDouble = *parser.getPathSpecificBins();

  // additional cuts applied to this histogram, combination of general ones and path-specific ones
  std::vector<edm::ParameterSet> allCutsVector = histConfig.getParameter<std::vector<edm::ParameterSet>>("rangeCuts");
  std::vector<edm::ParameterSet> pathSpecificCutsVector = *parser.getPathSpecificCuts();
  allCutsVector.insert(allCutsVector.end(), pathSpecificCutsVector.begin(), pathSpecificCutsVector.end());
  VarRangeCutColl<HLTGenValObject> rangeCuts(allCutsVector);

  // getting the custom tag
  const std::string tag = *parser.getTag();

  // checking validity of vsVar
  if (!vsVarFunc) {
    throw cms::Exception("ConfigError") << " vsVar " << vsVar << " is giving null ptr (likely empty) in " << __FILE__
                                        << "," << __LINE__ << std::endl;
  }

  // converting bin edges to float
  std::vector<float> binLowEdges;
  binLowEdges.reserve(binLowEdgesDouble.size());
  for (double lowEdge : binLowEdgesDouble)
    binLowEdges.push_back(lowEdge);

  // name and title are systematically constructed

  // remove potential leading "-" (which denotes that that trigger is ignored)
  std::string filterName = filter_;
  if (filterName.rfind('-', 0) == 0)
    filterName.erase(0, 1);

  std::string histName, histTitle;
  if (filter_ == "beforeAnyFilter") {  // this handles the naming of the "before" hist
    histName = objType_ + separator_ + path_ + separator_ + "GEN" + separator_ + "vs" + vsVar;
    histTitle = objType_ + " " + path_ + " GEN vs " + vsVar;
    if (!tag.empty()) {
      histName += separator_ + tag;
      histTitle += " " + tag;
    }
  } else {  // naming of all regular hists
    histName = objType_ + separator_ + path_ + separator_ + filterName + separator_ + "vs" + vsVar;
    histTitle = objType_ + " " + path_ + " " + filterName + " vs" + vsVar;

    // appending the tag, in case it is filled
    if (!tag.empty()) {
      histName += separator_ + tag;
      histTitle += " " + tag;
    }
  }

  auto me = iBooker.book1D(
      histName.c_str(), histTitle.c_str(), binLowEdges.size() - 1, binLowEdges.data());  // booking MonitorElement

  std::unique_ptr<HLTGenValHist> hist;  // creating the hist object

  hist = std::make_unique<HLTGenValHist1D>(me->getTH1(), vsVar, vsVarFunc, rangeCuts);

  hists_.emplace_back(std::move(hist));
}

// booker function for 2D hists
void HLTGenValHistCollFilter::book2D(DQMStore::IBooker& iBooker, const edm::ParameterSet& histConfig2D) {
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

  // name and title are systematically constructed

  // remove potential leading "-" (which denotes that that trigger is ignored)
  std::string filterName = filter_;
  if (filterName.rfind('-', 0) == 0)
    filterName.erase(0, 1);

  std::string histName, histTitle;
  if (filter_ == "beforeAnyFilter") {
    histName = objType_ + separator_ + path_ + separator_ + "GEN" + separator_ + "2Dvs" + vsVarX + separator_ + vsVarY;
    histTitle = objType_ + " " + path_ + " GEN 2D vs " + vsVarX + " " + vsVarY;
  } else {
    histName = objType_ + separator_ + path_ + separator_ + filterName + separator_ + "2Dvs" + vsVarX + vsVarY;
    histTitle = objType_ + " " + path_ + " " + filterName + " 2D vs" + vsVarX + " " + vsVarY;
  }

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
