#include "DataFormats/PatCandidates/interface/Particle.h"
#include "TopQuarkAnalysis/TopJetCombination/plugins/TtFullLepHypKinSolution.h"


TtFullLepHypKinSolution::TtFullLepHypKinSolution(const edm::ParameterSet& cfg):
  TtFullLepHypothesis( cfg ),
  nus_         (cfg.getParameter<edm::InputTag>("Neutrinos"      )),
  nuBars_      (cfg.getParameter<edm::InputTag>("NeutrinoBars"   )),  
  solWeight_   (cfg.getParameter<edm::InputTag>("solutionWeight" )) 
{
}

TtFullLepHypKinSolution::~TtFullLepHypKinSolution() { }

void
TtFullLepHypKinSolution::buildHypo(edm::Event& evt,
			           const edm::Handle<std::vector<pat::Electron > >& elecs, 
			           const edm::Handle<std::vector<pat::Muon> >& mus, 
			           const edm::Handle<std::vector<pat::Jet> >& jets, 
			           const edm::Handle<std::vector<pat::MET> >& mets, 
			           std::vector<int>& match,
				   const unsigned int iComb)
{ 
  edm::Handle<std::vector<double> > solWeight;  
  edm::Handle<std::vector<std::vector<int> > >   idcsVec; 
  edm::Handle<std::vector<reco::LeafCandidate> > nus;
  edm::Handle<std::vector<reco::LeafCandidate> > nuBars;  
  
  evt.getByLabel(solWeight_,    solWeight);
  evt.getByLabel(particleIdcs_, idcsVec  );
  evt.getByLabel(nus_,          nus      );
  evt.getByLabel(nuBars_,       nuBars   );  

  if( (*solWeight)[iComb]<0 ){
    // create empty hypothesis if no solution exists
    return;
  }

  // -----------------------------------------------------
  // add jets
  // -----------------------------------------------------
  if( !jets->empty() )  
    setCandidate(jets, match[0], b_   );
  if( !jets->empty() ) 
    setCandidate(jets, match[1], bBar_);

  // -----------------------------------------------------
  // add leptons
  // -----------------------------------------------------    
  if( !elecs->empty() && match[2]>=0) 
    setCandidate(elecs,  match[2], leptonBar_); 

  if( !elecs->empty() && match[3]>=0)
    setCandidate(elecs,  match[3], lepton_);

  if( !mus->empty() && match[4]>=0 && match[2]<0)
    setCandidate(mus,  match[4], leptonBar_);  

  // this 'else' happens if you have a wrong charge electron-muon-
  // solution so the indices are (b-idx, bbar-idx, 0, -1, 0, -1)
  // so the mu^+ is stored as l^-
  else if( !mus->empty() && match[4]>=0)
    setCandidate(mus,  match[4], lepton_);
    
  if( !mus->empty()  && match[5]>=0 && match[3]<0) 
    setCandidate(mus,  match[5], lepton_);   

  // this 'else' happens if you have a wrong charge electron-muon-
  // solution so the indices are (b-idx, bbar-idx, -1, 0, -1, 0)  
  // so the mu^- is stored as l^+
  else if( !mus->empty()  && match[5]>=0) 
    setCandidate(mus,  match[5], leptonBar_);   
      
  // -----------------------------------------------------
  // add neutrinos
  // -----------------------------------------------------
  if( !nus->empty() )
    setCandidate(nus,    iComb, neutrino_);

  if( !nuBars->empty() ) 
    setCandidate(nuBars, iComb, neutrinoBar_);   

}
