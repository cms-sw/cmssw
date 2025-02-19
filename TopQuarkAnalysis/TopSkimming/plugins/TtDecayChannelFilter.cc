#include "FWCore/Framework/interface/MakerMacros.h"
#include "TopQuarkAnalysis/TopSkimming/interface/TopDecayChannelFilter.h"
#include "TopQuarkAnalysis/TopSkimming/interface/TtDecayChannelSelector.h"

typedef TopDecayChannelFilter<TtDecayChannelSelector> 
                              TtDecayChannelFilter;

DEFINE_FWK_MODULE( TtDecayChannelFilter );
