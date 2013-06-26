#include "PhysicsTools/JetMCUtils/interface/combination.h"

#include "TopQuarkAnalysis/TopEventSelection/plugins/TtFullHadSignalSelMVAComputer.h"
#include "TopQuarkAnalysis/TopEventSelection/interface/TtFullHadSignalSelEval.h"

#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/PatCandidates/interface/Flags.h"


TtFullHadSignalSelMVAComputer::TtFullHadSignalSelMVAComputer(const edm::ParameterSet& cfg):
  jets_    (cfg.getParameter<edm::InputTag>("jets"))
{
  produces< double >("DiscSel");
}

  

TtFullHadSignalSelMVAComputer::~TtFullHadSignalSelMVAComputer()
{
}

void
TtFullHadSignalSelMVAComputer::produce(edm::Event& evt, const edm::EventSetup& setup)
{
  std::auto_ptr< double > pOutDisc (new double);
 
  mvaComputer.update<TtFullHadSignalSelMVARcd>(setup, "ttFullHadSignalSelMVA");

  // read name of the last processor in the MVA calibration
  // (to be used as meta information)
  edm::ESHandle<PhysicsTools::Calibration::MVAComputerContainer> calibContainer;
  setup.get<TtFullHadSignalSelMVARcd>().get( calibContainer );
  std::vector<PhysicsTools::Calibration::VarProcessor*> processors
    = (calibContainer->find("ttFullHadSignalSelMVA")).getProcessors();

  edm::Handle< std::vector<pat::Jet> > jets;
  evt.getByLabel(jets_, jets);
  
  //calculation of InputVariables
  //see TopQuarkAnalysis/TopTools/interface/TtFullHadSignalSel.h
  //                             /src/TtFullHadSignalSel.cc
  //all objects, jets, which are needed for the calculation
  //of the input-variables have to be passed to this class
  TtFullHadSignalSel selection(*jets);

  double discrim = evaluateTtFullHadSignalSel(mvaComputer, selection);

  *pOutDisc = discrim;
  
  evt.put(pOutDisc, "DiscSel");
  
  DiscSel = discrim;
}

void 
TtFullHadSignalSelMVAComputer::beginJob()
{
}

void 
TtFullHadSignalSelMVAComputer::endJob()
{
}

// implement the plugins for the computer container
// -> register TtFullHadSignalSelMVARcd
// -> define TtFullHadSignalSelMVAFileSource
MVA_COMPUTER_CONTAINER_IMPLEMENT(TtFullHadSignalSelMVA);
