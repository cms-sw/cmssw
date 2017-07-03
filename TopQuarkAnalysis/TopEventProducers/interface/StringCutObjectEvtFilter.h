#ifndef StringCutObjectEvtFilter_h
#define StringCutObjectEvtFilter_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

/**
   \class   StringCutObjectEvtFilter StringCutObjectEvtFilter.h "TopQuarkAnalysis/TopEventProducers/interface/StringCutObjectEvtFilter.h"

   \brief   Event filter based on the StringCutObjectSelector

   Template class to filter events based on member functions of a given object in the event
   and cuts that are parsed by the StringCutObjectSelector.

*/

template <typename T>
class StringCutObjectEvtFilter : public edm::EDFilter {

 public:

  /// default constructor
  explicit StringCutObjectEvtFilter(const edm::ParameterSet&);
  /// default destructor
  ~StringCutObjectEvtFilter() override{};

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
StringCutObjectEvtFilter<T>::StringCutObjectEvtFilter(const edm::ParameterSet& cfg):
  srcToken_(consumes<T>(cfg.getParameter<edm::InputTag>("src"))),
  cut_(cfg.getParameter<std::string>  ("cut"))
{}

template <typename T>
bool
StringCutObjectEvtFilter<T>::filter(edm::Event& evt, const edm::EventSetup& setup)
{
  edm::Handle<T> src;
  evt.getByToken(srcToken_, src);
  return cut_(*src);
}

#endif
