#include "DataFormats/PatCandidates/interface/Particle.h"
#include "TopQuarkAnalysis/TopJetCombination/plugins/TtFullHadHypKinFit.h"


TtFullHadHypKinFit::TtFullHadHypKinFit(const edm::ParameterSet& cfg):
  TtFullHadHypothesis( cfg ),
  status_      (cfg.getParameter<edm::InputTag>("status"      )),
  lightQTag_   (cfg.getParameter<edm::InputTag>("lightQTag"   )),
  lightQBarTag_(cfg.getParameter<edm::InputTag>("lightQBarTag")),
  bTag_        (cfg.getParameter<edm::InputTag>("bTag"        )),
  bBarTag_     (cfg.getParameter<edm::InputTag>("bBarTag"     )),
  lightPTag_   (cfg.getParameter<edm::InputTag>("lightPTag"   )),
  lightPBarTag_(cfg.getParameter<edm::InputTag>("lightPBarTag"))
{
}

TtFullHadHypKinFit::~TtFullHadHypKinFit() { }

void
TtFullHadHypKinFit::buildHypo(edm::Event& evt,
			      const edm::Handle<std::vector<pat::Jet> >& jets, 
			      std::vector<int>& match, const unsigned int iComb)
{
  edm::Handle<std::vector<int> > status;
  evt.getByLabel(status_, status);
  if( (*status)[iComb] != 0 ){
    // create empty hypothesis if kinematic fit did not converge
    return;
  }

  edm::Handle<std::vector<pat::Particle> > lightQ;
  edm::Handle<std::vector<pat::Particle> > lightQBar;
  edm::Handle<std::vector<pat::Particle> > b;
  edm::Handle<std::vector<pat::Particle> > bBar;
  edm::Handle<std::vector<pat::Particle> > lightP;
  edm::Handle<std::vector<pat::Particle> > lightPBar;

  evt.getByLabel(lightQTag_   , lightQ   );
  evt.getByLabel(lightQBarTag_, lightQBar);
  evt.getByLabel(bTag_        , b        );
  evt.getByLabel(bBarTag_     , bBar     );
  evt.getByLabel(lightPTag_   , lightP   );
  evt.getByLabel(lightPBarTag_, lightPBar);

  // -----------------------------------------------------
  // add jets
  // -----------------------------------------------------
  if( !( lightQ->empty() || lightQBar->empty() || b->empty() || bBar->empty() ||
	 lightP->empty() || lightPBar->empty() ) ) {
    setCandidate(lightQ   , iComb, lightQ_   );
    setCandidate(lightQBar, iComb, lightQBar_);
    setCandidate(b        , iComb, b_        );
    setCandidate(bBar     , iComb, bBar_     );
    setCandidate(lightP   , iComb, lightP_   );
    setCandidate(lightPBar, iComb, lightPBar_);
  }
}
