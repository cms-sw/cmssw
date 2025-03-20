#ifndef EmbeddingBeamSpotOnlineProducer_EmbeddingBeamSpotOnlineProducer_h
#define EmbeddingBeamSpotOnlineProducer_EmbeddingBeamSpotOnlineProducer_h

/**_________________________________________________________________
   class:   EmbeddingBeamSpotOnlineProducer.h
   package: TauAnalysis/MCEmbeddingTools
________________________________________________________________**/

#include "FWCore/Framework/interface/stream/EDProducer.h"

class EmbeddingBeamSpotOnlineProducer : public edm::stream::EDProducer<> {
public:
  /// constructor
  explicit EmbeddingBeamSpotOnlineProducer(const edm::ParameterSet &iConf);
  /// destructor
  ~EmbeddingBeamSpotOnlineProducer() override;

  /// produce a beam spot class
  void produce(edm::Event &iEvent, const edm::EventSetup &iSetup) override;

private:
  edm::InputTag beamSpotInput_;
};

#endif
