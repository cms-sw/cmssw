#include "PhysicsTools/JetMCUtils/interface/combination.h"

#include "TopQuarkAnalysis/TopEventSelection/plugins/TtSemiLepSignalSelMVAComputer.h"
#include "TopQuarkAnalysis/TopTools/interface/TtSemiLepEvtPartons.h"
#include "TopQuarkAnalysis/TopTools/interface/TtSemiLepSignalSelEval.h"

#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"

TtSemiLepSignalSelMVAComputer::TtSemiLepSignalSelMVAComputer(const edm::ParameterSet& cfg):
  leptons_ (cfg.getParameter<edm::InputTag>("leptons")),
  jets_    (cfg.getParameter<edm::InputTag>("jets")),
  METs_    (cfg.getParameter<edm::InputTag>("METs")),
  maxNJets_(cfg.getParameter<int>("maxNJets"))
{
  produces< double >("DiscSel");
}

TtSemiLepSignalSelMVAComputer::~TtSemiLepSignalSelMVAComputer()
{
}

void
TtSemiLepSignalSelMVAComputer::produce(edm::Event& evt, const edm::EventSetup& setup)
{
  std::auto_ptr< std::string >      pOutMeth (new std::string);
  std::auto_ptr< double >           pOutDisc (new double);

  mvaComputer.update<TtSemiLepSignalSelMVARcd>(setup, "ttSemiLepSignalSelMVA");

  // read name of the last processor in the MVA calibration
  // (to be used as meta information)
  edm::ESHandle<PhysicsTools::Calibration::MVAComputerContainer> calibContainer;
  setup.get<TtSemiLepSignalSelMVARcd>().get( calibContainer );
  std::vector<PhysicsTools::Calibration::VarProcessor*> processors
    = (calibContainer->find("ttSemiLepSignalSelMVA")).getProcessors();
  

  // get leptons, jets and MET
  edm::Handle< edm::View<reco::RecoCandidate> > leptons; 
  evt.getByLabel(leptons_, leptons);

  edm::Handle< std::vector<pat::Jet> > jet_handle;
  evt.getByLabel(jets_, jet_handle);
  const std::vector<pat::Jet> jets = *jet_handle;

  edm::Handle<edm::View<pat::MET> > MET_handle;
  evt.getByLabel(METs_,MET_handle);
  const edm::View<pat::MET> MET = *MET_handle;

  unsigned int nPartons = 4;

  // skip events with no appropriate lepton candidate in
  // or less jets than partons
  if( leptons->empty() || jets.size() < nPartons ) {
    for(unsigned int i = 0; i < nPartons; ++i) 
    *pOutDisc = 0.;
    evt.put(pOutDisc, "DiscSel");
    return;
  }

  math::XYZTLorentzVector lepton = leptons->begin()->p4();

  TtSemiLepSignalSel selection(jets,lepton,MET,maxNJets_);

  double discrim = evaluateTtSemiLepSignalSel(mvaComputer, selection);

  *pOutDisc = discrim;
  evt.put(pOutDisc, "DiscSel");
}

void 
TtSemiLepSignalSelMVAComputer::beginJob(const edm::EventSetup&)
{
}

void 
TtSemiLepSignalSelMVAComputer::endJob()
{
}

// implement the plugins for the computer container
// -> register TtSemiLepSignalSelMVARcd
// -> define TtSemiLepSignalSelMVAFileSource
MVA_COMPUTER_CONTAINER_IMPLEMENT(TtSemiLepSignalSelMVA);
