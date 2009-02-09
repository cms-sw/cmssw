#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiLepHypKinFit.h"

#include "DataFormats/PatCandidates/interface/Particle.h"

TtSemiLepHypKinFit::TtSemiLepHypKinFit(const edm::ParameterSet& cfg):
  TtSemiLepHypothesis( cfg ),
  status_     (cfg.getParameter<edm::InputTag>("status"     )),
  partonsHadP_(cfg.getParameter<edm::InputTag>("partonsHadP")),
  partonsHadQ_(cfg.getParameter<edm::InputTag>("partonsHadQ")),
  partonsHadB_(cfg.getParameter<edm::InputTag>("partonsHadB")),
  partonsLepB_(cfg.getParameter<edm::InputTag>("partonsLepB")),
  leptons_    (cfg.getParameter<edm::InputTag>("leptons"    )),
  neutrinos_  (cfg.getParameter<edm::InputTag>("neutrinos"  ))
{
}

TtSemiLepHypKinFit::~TtSemiLepHypKinFit() { }

void
TtSemiLepHypKinFit::buildHypo(edm::Event& evt,
			      const edm::Handle<edm::View<reco::RecoCandidate> >& leps, 
			      const edm::Handle<std::vector<pat::MET> >& mets, 
			      const edm::Handle<std::vector<pat::Jet> >& jets, 
			      std::vector<int>& match, const unsigned int iComb)
{
  edm::Handle<std::vector<int> > status;
  evt.getByLabel(status_, status);
  if( (*status)[iComb] != 0 ){
    // create empty hypothesis if kinematic fit did not converge
    return;
  }

  edm::Handle<std::vector<pat::Particle> > partonsHadP;
  edm::Handle<std::vector<pat::Particle> > partonsHadQ;
  edm::Handle<std::vector<pat::Particle> > partonsHadB;
  edm::Handle<std::vector<pat::Particle> > partonsLepB;
  edm::Handle<std::vector<pat::Particle> > leptons;
  edm::Handle<std::vector<pat::Particle> > neutrinos;

  evt.getByLabel(partonsHadP_,   partonsHadP);
  evt.getByLabel(partonsHadQ_,   partonsHadQ);
  evt.getByLabel(partonsHadB_,   partonsHadB);
  evt.getByLabel(partonsLepB_,   partonsLepB);
  evt.getByLabel(leptons_    ,   leptons    );
  evt.getByLabel(neutrinos_  ,   neutrinos  );

  // -----------------------------------------------------
  // add jets
  // -----------------------------------------------------
  if( !( partonsHadP->empty() || partonsHadQ->empty() ||
	 partonsHadB->empty() || partonsLepB->empty() ) ) {
    setCandidate(partonsHadP, iComb, lightQ_   );
    setCandidate(partonsHadQ, iComb, lightQBar_);
    setCandidate(partonsHadB, iComb, hadronicB_);
    setCandidate(partonsLepB, iComb, leptonicB_);
  }

  // -----------------------------------------------------
  // add lepton
  // -----------------------------------------------------
  if( !leptons->empty() )
    setCandidate(leptons, iComb, lepton_);
  match.push_back( 0 );
  
  // -----------------------------------------------------
  // add neutrino
  // -----------------------------------------------------
  if( !neutrinos->empty() )
    setCandidate(neutrinos, iComb, neutrino_);
}
