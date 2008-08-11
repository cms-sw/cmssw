#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiLepKinFit.h"
#include "TopQuarkAnalysis/TopTools/interface/TtSemiEvtPartons.h"

#include "DataFormats/PatCandidates/interface/Particle.h"

TtSemiLepKinFit::TtSemiLepKinFit(const edm::ParameterSet& cfg):
  TtSemiLepHypothesis( cfg ),
  status_   (cfg.getParameter<edm::InputTag>("status")),
  partons_  (cfg.getParameter<edm::InputTag>("partons")),
  leptons_  (cfg.getParameter<edm::InputTag>("leptons")),
  neutrinos_(cfg.getParameter<edm::InputTag>("neutrinos"))
{
}

TtSemiLepKinFit::~TtSemiLepKinFit() { }

void
TtSemiLepKinFit::buildHypo(edm::Event& evt,
				  const edm::Handle<edm::View<reco::RecoCandidate> >& leps, 
				  const edm::Handle<std::vector<pat::MET> >& mets, 
				  const edm::Handle<std::vector<pat::Jet> >& jets, 
				  std::vector<int>& match)
{
  edm::Handle<int> status;
  evt.getByLabel(status_, status);
  if(*status!=0){
    // create empty hypothesis if kinematic fit did not converge
    return;
  }

  edm::Handle<std::vector<pat::Particle> > partons;
  edm::Handle<std::vector<pat::Particle> > leptons;
  edm::Handle<std::vector<pat::Particle> > neutrinos;
  evt.getByLabel(partons_,   partons);
  evt.getByLabel(leptons_,   leptons);
  evt.getByLabel(neutrinos_, neutrinos);

  // -----------------------------------------------------
  // add jets
  // -----------------------------------------------------
  if( !(partons->size()<4) ) {
    setCandidate(partons, TtSemiEvtPartons::LightQ,    lightQ_   );
    setCandidate(partons, TtSemiEvtPartons::LightQBar, lightQBar_);
    setCandidate(partons, TtSemiEvtPartons::HadB,      hadronicB_);
    setCandidate(partons, TtSemiEvtPartons::LepB,      leptonicB_);
  }

  // -----------------------------------------------------
  // add lepton
  // -----------------------------------------------------
  if( !leptons->empty() )
    setCandidate(leptons, 0, lepton_);
  
  // -----------------------------------------------------
  // add neutrino
  // -----------------------------------------------------
  if( !neutrinos->empty() )
    setCandidate(neutrinos, 0, neutrino_);
}
