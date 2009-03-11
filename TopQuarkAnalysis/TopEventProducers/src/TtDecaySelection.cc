#include "TopQuarkAnalysis/TopEventProducers/interface/TtDecaySelection.h"

TtDecaySelection::TtDecaySelection(const edm::ParameterSet& cfg):
  src_( cfg.getParameter<edm::InputTag>( "src" ) ),
  sel_( cfg )
{
}

TtDecaySelection::~TtDecaySelection()
{
}

bool TtDecaySelection::filter(edm::Event& evt, const edm::EventSetup& setup)
{
  edm::Handle<TtGenEvent> genEvt;
  evt.getByLabel( src_, genEvt );
  return sel_( genEvt->particles(), src_.label() );
}
