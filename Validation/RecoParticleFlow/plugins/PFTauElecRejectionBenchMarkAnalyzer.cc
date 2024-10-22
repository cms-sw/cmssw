// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "RecoParticleFlow/Benchmark/interface/PFTauElecRejectionBenchmark.h"

using namespace edm;
using namespace reco;
using namespace std;

//
// class declaration

class PFTauElecRejectionBenchmarkAnalyzer : public edm::one::EDAnalyzer<> {
public:
  typedef dqm::legacy::DQMStore DQMStore;
  typedef dqm::legacy::MonitorElement MonitorElement;

  explicit PFTauElecRejectionBenchmarkAnalyzer(const edm::ParameterSet &);
  ~PFTauElecRejectionBenchmarkAnalyzer() override;

private:
  void beginJob() override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void endJob() override;
  // ----------member data ---------------------------

  string outputfile;
  DQMStore *db;
  string benchmarkLabel;
  double maxDeltaR;
  double minMCPt;
  double maxMCAbsEta;
  double minRecoPt;
  double maxRecoAbsEta;
  bool applyEcalCrackCut;
  string sGenMatchObjectLabel;

  edm::EDGetTokenT<edm::HepMCProduct> sGenParticleSource_tok_;
  edm::EDGetTokenT<reco::PFTauCollection> pfTauProducer_tok_;
  edm::EDGetTokenT<reco::PFTauDiscriminator> pfTauDiscriminatorByIsolationProducer_tok_;
  edm::EDGetTokenT<reco::PFTauDiscriminator> pfTauDiscriminatorAgainstElectronProducer_tok_;

  PFTauElecRejectionBenchmark PFTauElecRejectionBenchmark_;
};
/// PFTauElecRejection Benchmark

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
PFTauElecRejectionBenchmarkAnalyzer::PFTauElecRejectionBenchmarkAnalyzer(const edm::ParameterSet &iConfig)

{
  // now do what ever initialization is needed
  outputfile = iConfig.getUntrackedParameter<string>("OutputFile");
  benchmarkLabel = iConfig.getParameter<string>("BenchmarkLabel");
  sGenParticleSource_tok_ = consumes<edm::HepMCProduct>(iConfig.getParameter<InputTag>("InputTruthLabel"));
  maxDeltaR = iConfig.getParameter<double>("maxDeltaR");
  minMCPt = iConfig.getParameter<double>("minMCPt");
  maxMCAbsEta = iConfig.getParameter<double>("maxMCAbsEta");
  minRecoPt = iConfig.getParameter<double>("minRecoPt");
  maxRecoAbsEta = iConfig.getParameter<double>("maxRecoAbsEta");
  pfTauProducer_tok_ = consumes<reco::PFTauCollection>(iConfig.getParameter<InputTag>("PFTauProducer"));
  pfTauDiscriminatorByIsolationProducer_tok_ =
      consumes<reco::PFTauDiscriminator>(iConfig.getParameter<InputTag>("PFTauDiscriminatorByIsolationProducer"));
  pfTauDiscriminatorAgainstElectronProducer_tok_ =
      consumes<reco::PFTauDiscriminator>(iConfig.getParameter<InputTag>("PFTauDiscriminatorAgainstElectronProducer"));
  sGenMatchObjectLabel = iConfig.getParameter<string>("GenMatchObjectLabel");
  applyEcalCrackCut = iConfig.getParameter<bool>("ApplyEcalCrackCut");

  db = edm::Service<DQMStore>().operator->();

  PFTauElecRejectionBenchmark_.setup(outputfile,
                                     benchmarkLabel,
                                     maxDeltaR,
                                     minRecoPt,
                                     maxRecoAbsEta,
                                     minMCPt,
                                     maxMCAbsEta,
                                     sGenMatchObjectLabel,
                                     applyEcalCrackCut,
                                     db);
}

PFTauElecRejectionBenchmarkAnalyzer::~PFTauElecRejectionBenchmarkAnalyzer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
void PFTauElecRejectionBenchmarkAnalyzer::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  // get gen products
  Handle<HepMCProduct> mcevt;
  iEvent.getByToken(sGenParticleSource_tok_, mcevt);

  // get pftau collection
  Handle<PFTauCollection> thePFTau;
  iEvent.getByToken(pfTauProducer_tok_, thePFTau);

  // get iso discriminator association vector
  Handle<PFTauDiscriminator> thePFTauDiscriminatorByIsolation;
  iEvent.getByToken(pfTauDiscriminatorByIsolationProducer_tok_, thePFTauDiscriminatorByIsolation);

  // get anti-elec discriminator association vector
  Handle<PFTauDiscriminator> thePFTauDiscriminatorAgainstElectron;
  iEvent.getByToken(pfTauDiscriminatorAgainstElectronProducer_tok_, thePFTauDiscriminatorAgainstElectron);

  PFTauElecRejectionBenchmark_.process(
      mcevt, thePFTau, thePFTauDiscriminatorByIsolation, thePFTauDiscriminatorAgainstElectron);
}

// ------------ method called once each job just before starting event loop
// ------------
void PFTauElecRejectionBenchmarkAnalyzer::beginJob() {}

// ------------ method called once each job just after ending the event loop
// ------------
void PFTauElecRejectionBenchmarkAnalyzer::endJob() { PFTauElecRejectionBenchmark_.write(); }

// define this as a plug-in
DEFINE_FWK_MODULE(PFTauElecRejectionBenchmarkAnalyzer);
