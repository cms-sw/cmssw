#ifndef TtEvtFilter_h
#define TtEvtFilter_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "AnalysisDataFormats/TopObjects/interface/TtEvent.h"

/**
   \class   TtEvtFilter TtEvtFilter.h "TopQuarkAnalysis/TopEventProducers/interface/TtEvtFilter.h"

   \brief   Event filter based on the TtEvent structure

   Template class to filter events based on TtEvent objects.
   Member functions of
   
   * TtSemiLeptonicEvent
   * TtFullLeptonicEvent
   * TtFullHadronicEvent

   objects can be used in cuts that are parsed by the StringCutObjectSelector.
   
*/

template <typename T>
class TtEvtFilter : public edm::EDFilter {

 public:

  /// default constructor
  explicit TtEvtFilter(const edm::ParameterSet&);
  /// default destructor
  ~TtEvtFilter(){};
  
 private:

  /// filter function
  virtual bool filter(edm::Event&, const edm::EventSetup&);

 private:

  /// TtEvent input object
  edm::InputTag src_;
  /// cut string for event selection
  StringCutObjectSelector<T> cut_;
};

template <typename T>
TtEvtFilter<T>::TtEvtFilter(const edm::ParameterSet& cfg):
  src_(cfg.getParameter<edm::InputTag>("src")),
  cut_(cfg.getParameter<std::string>  ("cut"))
{}

template <typename T>
bool
TtEvtFilter<T>::filter(edm::Event& evt, const edm::EventSetup& setup)
{
  edm::Handle<T> src; 
  evt.getByLabel(src_, src);
  return cut_(*src);
}

#endif
