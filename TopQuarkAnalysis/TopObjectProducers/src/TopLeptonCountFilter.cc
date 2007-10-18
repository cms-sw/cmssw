//
// $Id: TopLeptonCountFilter.cc,v 1.2 2007/10/07 15:49:52 lowette Exp $
//

#include "TopQuarkAnalysis/TopObjectProducers/interface/TopLeptonCountFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "AnalysisDataFormats/TopObjects/interface/TopElectron.h"
#include "AnalysisDataFormats/TopObjects/interface/TopMuon.h"
#include "AnalysisDataFormats/TopObjects/interface/TopTau.h"


TopLeptonCountFilter::TopLeptonCountFilter(const edm::ParameterSet & iConfig) {
  electronSource_ = iConfig.getParameter<edm::InputTag>( "electronSource" );
  muonSource_     = iConfig.getParameter<edm::InputTag>( "muonSource" );
  tauSource_      = iConfig.getParameter<edm::InputTag>( "tauSource" );
  countElectrons_ = iConfig.getParameter<bool>         ( "countElectrons" );
  countMuons_     = iConfig.getParameter<bool>         ( "countMuons" );
  countTaus_      = iConfig.getParameter<bool>         ( "countTaus" );
  minNumber_      = iConfig.getParameter<unsigned int> ( "minNumber" );
}


TopLeptonCountFilter::~TopLeptonCountFilter() {
}


bool TopLeptonCountFilter::filter(edm::Event & iEvent, const edm::EventSetup & iSetup) {
  edm::Handle<std::vector<TopElectron> > electrons;
  if (countElectrons_) iEvent.getByLabel(electronSource_, electrons);
  edm::Handle<std::vector<TopMuon> > muons;
  if (countMuons_) iEvent.getByLabel(muonSource_, muons);
  edm::Handle<std::vector<TopTau> > taus;
  if (countTaus_) iEvent.getByLabel(tauSource_, taus);
  unsigned int nrLeptons = 0;
  nrLeptons += (countElectrons_ ? electrons->size() : 0);
  nrLeptons += (countMuons_     ? muons->size()     : 0);
  nrLeptons += (countTaus_      ? taus->size()      : 0);
  return nrLeptons >= minNumber_;
}
