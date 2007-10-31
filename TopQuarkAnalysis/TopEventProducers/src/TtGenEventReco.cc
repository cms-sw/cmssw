#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/TopDecaySubset.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/TtGenEventReco.h"

using namespace std;
using namespace reco;

TtGenEventReco::TtGenEventReco(const edm::ParameterSet& cfg):
  src_ ( cfg.getParameter<edm::InputTag>( "src"  ) ),
  init_( cfg.getParameter<edm::InputTag>( "init" ) )
{
  produces<TtGenEvent>();
}

TtGenEventReco::~TtGenEventReco()
{
}

void
TtGenEventReco::produce(edm::Event& evt, const edm::EventSetup& setup)
{     
  edm::Handle<reco::CandidateCollection> parts;
  evt.getByLabel(src_,  parts);

  edm::Handle<reco::CandidateCollection> inits;
  evt.getByLabel(init_, inits);

  //add TopDecayTree
  reco::CandidateRefProd cands( parts );

  //add InitialStatePartons
  reco::CandidateRefProd initParts( inits );

  //add genEvt to the output stream
  TtGenEvent* genEvt = new TtGenEvent( cands, initParts );
  std::auto_ptr<TtGenEvent> gen( genEvt );
  evt.put( gen );
}
