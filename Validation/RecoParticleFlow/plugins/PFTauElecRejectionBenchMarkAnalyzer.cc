// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

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

 
class PFTauElecRejectionBenchmarkAnalyzer : public edm::EDAnalyzer {
public:
  explicit PFTauElecRejectionBenchmarkAnalyzer(const edm::ParameterSet&);
  ~PFTauElecRejectionBenchmarkAnalyzer();


private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;  
  // ----------member data ---------------------------

  string outputfile;
  DQMStore * db;
  string benchmarkLabel;
  InputTag sGenParticleSource;
  double maxDeltaR;
  double minMCPt;
  double maxMCAbsEta;
  double minRecoPt;
  double maxRecoAbsEta;
  InputTag pfTauProducer; 
  InputTag pfTauDiscriminatorByIsolationProducer; 
  InputTag pfTauDiscriminatorAgainstElectronProducer; 
  bool applyEcalCrackCut;
  string  sGenMatchObjectLabel;
  
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
PFTauElecRejectionBenchmarkAnalyzer::PFTauElecRejectionBenchmarkAnalyzer(const edm::ParameterSet& iConfig)

{
  //now do what ever initialization is needed
  outputfile = 
    iConfig.getUntrackedParameter<string>("OutputFile");
  benchmarkLabel =
    iConfig.getParameter<string>("BenchmarkLabel");
  sGenParticleSource = 
    iConfig.getParameter<InputTag>("InputTruthLabel");
  maxDeltaR = 
    iConfig.getParameter<double>("maxDeltaR");	  
  minMCPt  = 
    iConfig.getParameter<double>("minMCPt"); 
  maxMCAbsEta = 
    iConfig.getParameter<double>("maxMCAbsEta"); 
  minRecoPt  = 
    iConfig.getParameter<double>("minRecoPt"); 
  maxRecoAbsEta = 
    iConfig.getParameter<double>("maxRecoAbsEta"); 
  pfTauProducer = 
    iConfig.getParameter<InputTag>("PFTauProducer");
  pfTauDiscriminatorByIsolationProducer = 
    iConfig.getParameter<InputTag>("PFTauDiscriminatorByIsolationProducer");
  pfTauDiscriminatorAgainstElectronProducer = 
    iConfig.getParameter<InputTag>("PFTauDiscriminatorAgainstElectronProducer");
  sGenMatchObjectLabel =
    iConfig.getParameter<string>("GenMatchObjectLabel");
  applyEcalCrackCut =
    iConfig.getParameter<bool>("ApplyEcalCrackCut");


  db = edm::Service<DQMStore>().operator->();
  

  PFTauElecRejectionBenchmark_.setup(
			outputfile,
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


PFTauElecRejectionBenchmarkAnalyzer::~PFTauElecRejectionBenchmarkAnalyzer()
{
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called to for each event  ------------
void
PFTauElecRejectionBenchmarkAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  // get gen products
  Handle<HepMCProduct> mcevt;
  iEvent.getByLabel(sGenParticleSource, mcevt);

  // get pftau collection
  Handle<PFTauCollection> thePFTau;
  iEvent.getByLabel(pfTauProducer,thePFTau);
  
  // get iso discriminator association vector
  Handle<PFTauDiscriminator> thePFTauDiscriminatorByIsolation;
  iEvent.getByLabel(pfTauDiscriminatorByIsolationProducer,thePFTauDiscriminatorByIsolation);

  // get anti-elec discriminator association vector
  Handle<PFTauDiscriminator> thePFTauDiscriminatorAgainstElectron;
  iEvent.getByLabel(pfTauDiscriminatorAgainstElectronProducer,thePFTauDiscriminatorAgainstElectron);

  PFTauElecRejectionBenchmark_.process(mcevt, thePFTau, thePFTauDiscriminatorByIsolation, 
				       thePFTauDiscriminatorAgainstElectron);
}


// ------------ method called once each job just before starting event loop  ------------
void 
PFTauElecRejectionBenchmarkAnalyzer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
PFTauElecRejectionBenchmarkAnalyzer::endJob() {
  PFTauElecRejectionBenchmark_.write();
}

//define this as a plug-in
DEFINE_FWK_MODULE(PFTauElecRejectionBenchmarkAnalyzer);

