#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

/**
   \class   StringCutObjectEvtFilter StringCutObjectEvtFilter.h "TopQuarkAnalysis/TopEventProducers/interface/StringCutObjectEvtFilter.h"

   \brief   Event filter based on the StringCutObjectSelector

   Template class to filter events based on member functions of a given object in the event
   and cuts that are parsed by the StringCutObjectSelector.

*/

template <typename T>
class StringCutObjectEvtFilter : public edm::stream::EDFilter<> {
public:
  /// default constructor
  explicit StringCutObjectEvtFilter(const edm::ParameterSet&);

private:
  /// filter function
  bool filter(edm::Event&, const edm::EventSetup&) override;

private:
  /// input object
  edm::EDGetTokenT<T> srcToken_;
  /// cut string for event selection
  StringCutObjectSelector<T> cut_;
};

template <typename T>
StringCutObjectEvtFilter<T>::StringCutObjectEvtFilter(const edm::ParameterSet& cfg)
    : srcToken_(consumes<T>(cfg.getParameter<edm::InputTag>("src"))), cut_(cfg.getParameter<std::string>("cut")) {}

template <typename T>
bool StringCutObjectEvtFilter<T>::filter(edm::Event& evt, const edm::EventSetup& setup) {
  return cut_(evt.get(srcToken_));
}

#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "AnalysisDataFormats/TopObjects/interface/TtFullHadronicEvent.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiLeptonicEvent.h"
#include "AnalysisDataFormats/TopObjects/interface/TtFullLeptonicEvent.h"

using TtGenEvtFilter = StringCutObjectEvtFilter<TtGenEvent>;
using TtFullHadEvtFilter = StringCutObjectEvtFilter<TtFullHadronicEvent>;
using TtFullLepEvtFilter = StringCutObjectEvtFilter<TtFullLeptonicEvent>;
using TtSemiLepEvtFilter = StringCutObjectEvtFilter<TtSemiLeptonicEvent>;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TtGenEvtFilter);
DEFINE_FWK_MODULE(TtFullHadEvtFilter);
DEFINE_FWK_MODULE(TtFullLepEvtFilter);
DEFINE_FWK_MODULE(TtSemiLepEvtFilter);
