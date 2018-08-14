#ifndef TtFullHadSignalSelMVAComputer_h
#define TtFullHadSignalSelMVAComputer_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "PhysicsTools/MVAComputer/interface/HelperMacros.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputerCache.h"

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

#ifndef TtFullHadSignalSelMVARcd_defined  // to avoid conflicts with the TopFullHadLepSignalSelMVATrainer
#define TtFullHadSignalSelMVARcd_defined
MVA_COMPUTER_CONTAINER_DEFINE(TtFullHadSignalSelMVA);  // defines TopFullHadLepSignalSelMVARcd
#endif

class TtFullHadSignalSelMVAComputer : public edm::EDProducer {

 public:

  explicit TtFullHadSignalSelMVAComputer(const edm::ParameterSet&);
  ~TtFullHadSignalSelMVAComputer() override;

 private:

  void beginJob() override;
  void produce(edm::Event& evt, const edm::EventSetup& setup) override;
  void endJob() override;

  edm::EDGetTokenT< std::vector<pat::Jet> > jetsToken_;

  PhysicsTools::MVAComputerCache mvaComputer;

  double DiscSel;

};

#endif
