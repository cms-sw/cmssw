/** \class DQMHistNormalizer
 *  
 *  Class to produce efficiency histograms by dividing nominator by denominator histograms
 *
 *  \author Christian Veelken, UC Davis
 */

// framework & common header files
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

//Regexp handling
#include "classlib/utils/RegexpMatch.h"
#include "classlib/utils/Regexp.h"

#include <string>
#include <vector>
#include <map>

using namespace std;

class DQMHistNormalizer : public edm::one::EDAnalyzer<edm::one::SharedResources, edm::one::WatchRuns> {
public:
  typedef dqm::legacy::DQMStore DQMStore;
  typedef dqm::legacy::MonitorElement MonitorElement;

  explicit DQMHistNormalizer(const edm::ParameterSet&);
  ~DQMHistNormalizer() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void beginRun(const edm::Run& r, const edm::EventSetup& c) override {}
  void endRun(const edm::Run& r, const edm::EventSetup& c) override;

private:
  lat::Regexp* buildRegex(const string& expr);
  vector<string> plotNamesToNormalize_;  //root name used by all the plots that must be normalized
  string reference_;
};

DQMHistNormalizer::DQMHistNormalizer(const edm::ParameterSet& cfg)
    : plotNamesToNormalize_(cfg.getParameter<std::vector<string> >("plotNamesToNormalize")),
      reference_(cfg.getParameter<string>("reference")) {
  usesResource("DQMStore");
  //std::cout << "<DQMHistNormalizer::DQMHistNormalizer>:" << std::endl;
}

DQMHistNormalizer::~DQMHistNormalizer() {
  //--- nothing to be done yet
}

void DQMHistNormalizer::analyze(const edm::Event&, const edm::EventSetup&) {
  //--- nothing to be done yet
}

lat::Regexp* DQMHistNormalizer::buildRegex(const string& expr) {
  lat::Regexp* rx = nullptr;
  try {
    rx = new lat::Regexp(expr, 0, lat::Regexp::Wildcard);
    rx->study();
  } catch (lat::Error& e) {
    throw cms::Exception("DQMHistNormalizer")
        << "Invalid regular expression '" << expr.c_str() << "':" << e.explain().c_str();
  }
  return rx;
}

void DQMHistNormalizer::endRun(const edm::Run& r, const edm::EventSetup& c) {
  //std::cout << "<DQMHistNormalizer::endJob>:" << std::endl;

  //--- check that DQMStore service is available
  if (!edm::Service<DQMStore>().isAvailable()) {
    edm::LogError("endJob") << " Failed to access dqmStore --> histograms will NOT be plotted !!";
    return;
  }

  DQMStore& dqmStore = (*edm::Service<DQMStore>());

  vector<MonitorElement*> allOurMEs = dqmStore.getAllContents("RecoTauV/");
  lat::Regexp* refregex = buildRegex("*RecoTauV/*/" + reference_);
  vector<lat::Regexp*> toNormRegex;
  for (std::vector<string>::const_iterator toNorm = plotNamesToNormalize_.begin();
       toNorm != plotNamesToNormalize_.end();
       ++toNorm)
    toNormRegex.push_back(buildRegex("*RecoTauV/*/" + *toNorm));

  map<string, MonitorElement*> refsMap;
  vector<MonitorElement*> toNormElements;

  for (vector<MonitorElement*>::const_iterator element = allOurMEs.begin(); element != allOurMEs.end(); ++element) {
    string pathname = (*element)->getFullname();
    //cout << pathname << endl;
    //Matches reference
    if (refregex->match(pathname)) {
      //cout << "Matched to ref" << endl;
      string dir = pathname.substr(0, pathname.rfind('/'));
      if (refsMap.find(dir) != refsMap.end()) {
        edm::LogInfo("DQMHistNormalizer")
            << "DQMHistNormalizer::endRun: Warning! found multiple normalizing references for dir: " << dir << "!";
        edm::LogInfo("DQMHistNormalizer") << "     " << (refsMap[dir])->getFullname();
        edm::LogInfo("DQMHistNormalizer") << "     " << pathname;
      } else {
        refsMap[dir] = *element;
      }
    }

    //Matches targets
    for (vector<lat::Regexp*>::const_iterator reg = toNormRegex.begin(); reg != toNormRegex.end(); ++reg) {
      if ((*reg)->match(pathname)) {
        //cout << "Matched to target" << endl;
        toNormElements.push_back(*element);
        //cout << "Filled the collection" << endl;
      }
    }
  }

  delete refregex;
  for (vector<lat::Regexp*>::const_iterator reg = toNormRegex.begin(); reg != toNormRegex.end(); ++reg)
    delete *reg;

  for (vector<MonitorElement*>::const_iterator matchingElement = toNormElements.begin();
       matchingElement != toNormElements.end();
       ++matchingElement) {
    string meName = (*matchingElement)->getFullname();
    string dir = meName.substr(0, meName.rfind('/'));

    if (refsMap.find(dir) == refsMap.end()) {
      edm::LogInfo("DQMHistNormalizer") << "DQMHistNormalizer::endRun: Error! normalizing references for " << meName
                                        << " not found! Skipping...";
      continue;
    }

    float norm = refsMap[dir]->getTH1()->GetEntries();
    TH1* hist = (*matchingElement)->getTH1();
    if (norm != 0.) {
      if (!hist->GetSumw2N())
        hist->Sumw2();
      hist->Scale(1 / norm);  //use option "width" to divide the bin contents and errors by the bin width?
    } else {
      edm::LogInfo("DQMHistNormalizer") << "DQMHistNormalizer::endRun: Error! Normalization failed in "
                                        << hist->GetTitle() << "!";
    }

  }  //    for(vector<MonitorElement *>::const_iterator matchingElement = matchingElemts.begin(); matchingElement = matchingElemts.end(); ++matchingElement)
}

DEFINE_FWK_MODULE(DQMHistNormalizer);
