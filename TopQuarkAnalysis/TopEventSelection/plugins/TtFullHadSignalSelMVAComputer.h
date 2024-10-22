#ifndef TtFullHadSignalSelMVAComputer_h
#define TtFullHadSignalSelMVAComputer_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "PhysicsTools/MVAComputer/interface/HelperMacros.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputerCache.h"

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

#ifndef TtFullHadSignalSelMVARcd_defined  // to avoid conflicts with the TopFullHadLepSignalSelMVATrainer
#define TtFullHadSignalSelMVARcd_defined
MVA_COMPUTER_CONTAINER_DEFINE(TtFullHadSignalSelMVA);  // defines TopFullHadLepSignalSelMVARcd
#endif

class TtFullHadSignalSelMVAComputer : public edm::stream::EDProducer<> {
public:
  explicit TtFullHadSignalSelMVAComputer(const edm::ParameterSet&);

private:
  void produce(edm::Event& evt, const edm::EventSetup& setup) override;

  edm::ESGetToken<PhysicsTools::Calibration::MVAComputerContainer, TtFullHadSignalSelMVARcd> mvaToken_;
  edm::EDGetTokenT<std::vector<pat::Jet> > jetsToken_;
  edm::EDPutTokenT<double> putToken_;

  PhysicsTools::MVAComputerCache mvaComputer;
};

#endif
