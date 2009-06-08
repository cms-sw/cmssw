#include "TopQuarkAnalysis/TopJetCombination/plugins/TtFullLepHypGenMatch.h"
#include "AnalysisDataFormats/TopObjects/interface/TtFullLepEvtPartons.h"
#include "DataFormats/Math/interface/deltaR.h"

TtFullLepHypGenMatch::TtFullLepHypGenMatch(const edm::ParameterSet& cfg):
  TtFullLepHypothesis( cfg ) 
{  
}

TtFullLepHypGenMatch::~TtFullLepHypGenMatch() { }

void
TtFullLepHypGenMatch::buildHypo(edm::Event& evt,
			        const edm::Handle<std::vector<pat::Electron > >& elecs, 
			        const edm::Handle<std::vector<pat::Muon> >& mus, 
			        const edm::Handle<std::vector<pat::Jet> >& jets, 
			        const edm::Handle<std::vector<pat::MET> >& mets, 
			        std::vector<int>& match,
				const unsigned int iComb)
{
  // -----------------------------------------------------
  // add jets
  // -----------------------------------------------------
  for(unsigned idx=0; idx<match.size(); ++idx){
    if( isValid(match[idx], jets) ){
      switch(idx){
      case TtFullLepEvtPartons::B:
	setCandidate(jets, match[idx], b_); break;
      case TtFullLepEvtPartons::BBar:
	setCandidate(jets, match[idx], bBar_); break;	
      }
    }
  }

  // -----------------------------------------------------
  // add leptons
  // -----------------------------------------------------
  // get genEvent
  edm::Handle<TtGenEvent> genEvt;
  evt.getByLabel("genEvt", genEvt);
  
  // push back fake indices if no leptons in genevent
  if( !genEvt->isFullLeptonic() || !genEvt->lepton() || !genEvt->leptonBar() ){
    match.push_back( -1 );
    match.push_back( -1 );
    match.push_back( -1 );
    match.push_back( -1 );          
  }  
  else if(genEvt->isFullLeptonic(WDecay::kElec, WDecay::kElec) && elecs->size()>=2){    
    //search indices for electrons
    int iLepBar = findMatchingLepton(genEvt->leptonBar(), elecs);
    setCandidate(elecs, iLepBar, leptonBar_);
    match.push_back( iLepBar );
    int iLep = findMatchingLepton(genEvt->lepton(), elecs);
    setCandidate(elecs, iLep, lepton_);
    match.push_back( iLep );
    
    // fake indices for muons  
    match.push_back( -1 );
    match.push_back( -1 );         
  }
  else if(genEvt->isFullLeptonic(WDecay::kElec, WDecay::kMuon) && !elecs->empty() && !mus->empty() ){
    if(genEvt->leptonBar()->isElectron()){       
      // push back index for e+
      int iLepBar = findMatchingLepton(genEvt->leptonBar(), elecs);
      setCandidate(elecs, iLepBar, leptonBar_);
      match.push_back( iLepBar );
      // push back fake indices for e- and mu+
      match.push_back( -1 );
      match.push_back( -1 ); 
      // push back index for mu-     
      int iLep = findMatchingLepton(genEvt->lepton(), mus);
      setCandidate(mus, iLep, lepton_);
      match.push_back( iLep );              
    }
    else{       
      // push back fake index for e+    
      match.push_back( -1 );  
      // push back index for e-  
      int iLepBar = findMatchingLepton(genEvt->leptonBar(), mus);
      setCandidate(mus, iLepBar, leptonBar_);
      match.push_back( iLepBar ); 
      // push back index for mu+
      int iLep = findMatchingLepton(genEvt->lepton(), elecs);
      setCandidate(elecs, iLep, lepton_);
      match.push_back( iLep );
      // push back fake index for mu-     
      match.push_back( -1 );              
    }
  }  
  else if(genEvt->isFullLeptonic(WDecay::kMuon, WDecay::kMuon) &&  mus->size()>=2 ){     
    // fake indices for electrons
    match.push_back( -1 );
    match.push_back( -1 );  
    
    //search indices for electrons
    int iLepBar = findMatchingLepton(genEvt->leptonBar(), mus);
    setCandidate(mus, iLepBar, leptonBar_);
    match.push_back( iLepBar );
    int iLep = findMatchingLepton(genEvt->lepton(), mus);
    setCandidate(mus, iLep, lepton_);
    match.push_back( iLep );    
  }
  else{ //this 'else' should happen if at least one genlepton is a tau
    match.push_back( -1 );
    match.push_back( -1 );
    match.push_back( -1 );
    match.push_back( -1 );   
  }

  // -----------------------------------------------------
  // add met and neutrinos
  // -----------------------------------------------------  
  if( !mets->empty() ){
    //setCandidate(mets, 0, met_);
    buildMatchingNeutrinos(evt, mets);  
  }    
}

template <typename O>
int
TtFullLepHypGenMatch::findMatchingLepton(const reco::GenParticle* genLep, const edm::Handle<std::vector<O> >& leps)
{
  int idx=-1;   
  double minDR = -1;
  for(unsigned i=0; i<leps->size(); ++i){
    double dR = deltaR(genLep->eta(), genLep->phi(), (*leps)[i].eta(), (*leps)[i].phi());
    if(minDR<0 || dR<minDR){
      minDR=dR;
      idx=i;
    }     
  }
  return idx;
}

void
TtFullLepHypGenMatch::buildMatchingNeutrinos(edm::Event& evt, const edm::Handle<std::vector<pat::MET> >& mets)
{
  // get genEvent
  edm::Handle<TtGenEvent> genEvt;
  evt.getByLabel("genEvt", genEvt);
  
  if( genEvt->isTtBar() && genEvt->isFullLeptonic() && genEvt->neutrino() && genEvt->neutrinoBar() ){
    double momXNu    = genEvt->neutrino()   ->px();
    double momYNu    = genEvt->neutrino()   ->py(); 
    double momXNuBar = genEvt->neutrinoBar()->px();
    double momYNuBar = genEvt->neutrinoBar()->py();
        
    double momXMet = mets->at(0).px();
    double momYMet = mets->at(0).py();

    double momXNeutrino = 0.5*(momXNu - momXNuBar + momXMet);
    double momYNeutrino = 0.5*(momYNu - momYNuBar + momYMet);   
    double momXNeutrinoBar = momXMet - momXNeutrino;
    double momYNeutrinoBar = momYMet - momYNeutrino; 
  
    math::XYZTLorentzVector recNuFM(momXNeutrino,momYNeutrino,0,sqrt(momXNeutrino * momXNeutrino + momYNeutrino * momYNeutrino));    
    recNu = new reco::LeafCandidate(0,recNuFM);
    
    math::XYZTLorentzVector recNuBarFM(momXNeutrinoBar,momYNeutrinoBar,0,sqrt(momXNeutrinoBar * momXNeutrinoBar + momYNeutrinoBar * momYNeutrinoBar));
    recNuBar = new reco::LeafCandidate(0,recNuBarFM);		      
  }
}
