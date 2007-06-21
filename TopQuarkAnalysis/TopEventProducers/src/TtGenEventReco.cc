#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Candidate/interface/ShallowCloneCandidate.h"

#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/TtGenEventReco.h"

using namespace std;
using namespace reco;

TtGenEventReco::TtGenEventReco(const edm::ParameterSet& cfg):
  src_ ( cfg.getParameter<edm::InputTag>( "src" ) )
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
  evt.getByLabel(src_, parts);

  reco::CandidateCollection cands;
  reco::CandidateCollection::const_iterator part = parts->begin(); 
  for( int idx=0; part!=parts->end(); ++part, ++idx ) {
    CandidateBaseRef ref( CandidateRef( parts, idx ) );
    cands.push_back( new reco::ShallowCloneCandidate( ref ) );
  }
  
  TtGenEvent* genEvt = new TtGenEvent( cands );
  std::auto_ptr<TtGenEvent> gen( genEvt );

  evt.put( gen );
}
