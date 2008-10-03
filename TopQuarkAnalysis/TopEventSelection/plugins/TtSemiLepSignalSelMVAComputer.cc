#include "PhysicsTools/JetMCUtils/interface/combination.h"

#include "TopQuarkAnalysis/TopEventSelection/plugins/TtSemiLepSignalSelMVAComputer.h"
#include "TopQuarkAnalysis/TopTools/interface/TtSemiLepEvtPartons.h"
#include "TopQuarkAnalysis/TopTools/interface/TtSemiLepSignalSelEval.h"

#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include <DataFormats/PatCandidates/interface/Muon.h>

TtSemiLepSignalSelMVAComputer::TtSemiLepSignalSelMVAComputer(const edm::ParameterSet& cfg):
  leptons_ (cfg.getParameter<edm::InputTag>("leptons")),
  jets_    (cfg.getParameter<edm::InputTag>("jets")),
  METs_    (cfg.getParameter<edm::InputTag>("METs")),
  nJetsMax_(cfg.getParameter<int>("nJetsMax"))
{
  produces< double        >("DiscSel");
}

  

TtSemiLepSignalSelMVAComputer::~TtSemiLepSignalSelMVAComputer()
{
}

void
TtSemiLepSignalSelMVAComputer::produce(edm::Event& evt, const edm::EventSetup& setup)
{
  
  std::auto_ptr< double >        pOutDisc (new double);
  
  mvaComputer.update<TtSemiLepSignalSelMVARcd>(setup, "ttSemiLepSignalSelMVA");

  // read name of the last processor in the MVA calibration
  // (to be used as meta information)
  edm::ESHandle<PhysicsTools::Calibration::MVAComputerContainer> calibContainer;
  setup.get<TtSemiLepSignalSelMVARcd>().get( calibContainer );
  std::vector<PhysicsTools::Calibration::VarProcessor*> processors
    = (calibContainer->find("ttSemiLepSignalSelMVA")).getProcessors();


  edm::Handle<edm::View<pat::MET> > MET_handle;
  evt.getByLabel(METs_,MET_handle);
  if(!MET_handle.isValid()) return;
  const edm::View<pat::MET> MET = *MET_handle;

  edm::Handle< edm::View<pat::Muon> > lepton_handle; 
  evt.getByLabel(leptons_, lepton_handle);
  if(!lepton_handle.isValid()) return;
  const edm::View<pat::Muon>& leptons = *lepton_handle;
  int nleptons = 0;
  for(edm::View<pat::Muon>::const_iterator it = leptons.begin(); it!=leptons.end(); it++) {
    if(it->pt()>30 && fabs(it->eta())<2.1) nleptons++;
  }
  
  math::XYZTLorentzVector lepton = leptons.begin()->p4();

  edm::Handle< std::vector<pat::Jet> > jet_handle;
  evt.getByLabel(jets_, jet_handle);
  if(!jet_handle.isValid()) return;
  const std::vector<pat::Jet> jets = *jet_handle;
  //std::sort(jets.begin(),jets.end(),JetETComparison);
  
  double dRmin = 9999.;
  std::vector<pat::Jet> seljets;
  for(std::vector<pat::Jet>::const_iterator it = jets.begin(); it != jets.end(); it++) {
    if(it->et()>20. && fabs(it->eta())<2.4) {
      math::XYZTLorentzVector tv = it->p4();
      double tmpdR = TMath::Sqrt((tv.Eta()-lepton.Eta())*(tv.Eta()-lepton.Eta())
	    		        +(tv.Phi()-lepton.Phi())*(tv.Phi()-lepton.Phi()));
      if(tmpdR<dRmin) dRmin = tmpdR;
      seljets.push_back(*it);
    }
  }
  
  unsigned int nPartons = 4;
  double discrim;

  // skip events with no appropriate lepton candidate in
  if( nleptons!=1                    ||
      seljets.size() < nPartons      ||
      leptons.begin()->caloIso()>=1  ||
      leptons.begin()->trackIso()>=3 ||
      jets.begin()->et()<=65.        ||
      dRmin<=0.3 ) discrim = -1.;
  else {
    math::XYZTLorentzVector lepton = leptons.begin()->p4();

    TtSemiLepSignalSel selection(jets,lepton,MET,nJetsMax_);

    discrim = evaluateTtSemiLepSignalSel(mvaComputer, selection);
  }

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
