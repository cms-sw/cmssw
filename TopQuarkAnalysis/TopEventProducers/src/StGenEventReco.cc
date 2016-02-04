#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "AnalysisDataFormats/TopObjects/interface/StGenEvent.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/TopDecaySubset.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/StGenEventReco.h"

StGenEventReco::StGenEventReco(const edm::ParameterSet& cfg):
  src_ ( cfg.getParameter<edm::InputTag>( "src"  ) ),
  init_( cfg.getParameter<edm::InputTag>( "init" ) )
{
  produces<StGenEvent>();
}

StGenEventReco::~StGenEventReco()
{
}

void
StGenEventReco::produce(edm::Event& evt, const edm::EventSetup& setup)
{     
  edm::Handle<reco::GenParticleCollection> parts;
  evt.getByLabel(src_,  parts);

  edm::Handle<reco::GenParticleCollection> inits;
  evt.getByLabel(init_, inits);

  //add TopDecayTree
  reco::GenParticleRefProd cands( parts );

  //add InitialStatePartons
  reco::GenParticleRefProd initParts( inits );

  //add genEvt to the output stream
  StGenEvent* genEvt = new StGenEvent( cands, initParts );
  std::auto_ptr<StGenEvent> gen( genEvt );
  evt.put( gen );
}
