#include "SimGeneral/PreMixingModule/interface/PreMixingDigiSimLinkWorker.h"
#include "SimGeneral/PreMixingModule/interface/PreMixingWorkerFactory.h"
#include "SimDataFormats/DigiSimLinks/interface/DTDigiSimLinkCollection.h"

// DT does not use DetSetVector, so have to specialize put()
template <>
void PreMixingDigiSimLinkWorker<DTDigiSimLinkCollection>::addPileups(PileUpEventPrincipal const& pep, edm::EventSetup const& iSetup) {
  edm::Handle<DTDigiSimLinkCollection> digis;
  pep.getByLabel(pileupTag_, digis);
  if(digis.isValid()) {
    for(const auto& elem: *digis) {
      merged_->put(elem.second, elem.first);
    }
  }
}

using PreMixingDTDigiSimLinkWorker = PreMixingDigiSimLinkWorker<DTDigiSimLinkCollection>;

DEFINE_EDM_PLUGIN(PreMixingWorkerFactory, PreMixingDTDigiSimLinkWorker , "PreMixingDTDigiSimLinkWorker");
