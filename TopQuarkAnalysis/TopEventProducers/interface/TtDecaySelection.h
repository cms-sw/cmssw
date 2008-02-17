#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "TopQuarkAnalysis/TopSkimming/interface/TtDecayChannelSelector.h"

class TtDecaySelection : public edm::EDFilter {
 public:

  explicit TtDecaySelection(const edm::ParameterSet&);
  ~TtDecaySelection();
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  
 private:

  edm::InputTag src_;    
  TtDecayChannelSelector sel_;
};
