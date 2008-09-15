#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "TopQuarkAnalysis/Examples/plugins/TopElecAnalyzer.h"


TopElecAnalyzer::TopElecAnalyzer(const edm::ParameterSet& cfg):
  elecs_(cfg.getParameter<edm::InputTag>("input"))
{
  edm::Service<TFileService> fs;
  
  NrElec_ = fs->make<TH1I>("NrElec",  "Num_{Elecs}",    10,  0 , 10 );
  ptElec_ = fs->make<TH1F>("ptElec",  "pt_{Elecs}",    100,  0.,300.);
  enElec_ = fs->make<TH1F>("enElec",  "energy_{Elecs}",100,  0.,300.);
  etaElec_= fs->make<TH1F>("etaElec", "eta_{Elecs}",   100, -3.,  3.);
  phiElec_= fs->make<TH1F>("phiElec", "phi_{Elecs}",   100, -5.,  5.);
  trigMatchElec_= fs->make<TH1F>("trigMatchElec", "trigMatch_{Elec}", 100, -5.,  5.);
}

TopElecAnalyzer::~TopElecAnalyzer()
{
}

void
TopElecAnalyzer::analyze(const edm::Event& evt, const edm::EventSetup& setup)
{       
  edm::Handle<std::vector<pat::Electron> > elecs;
  evt.getByLabel(elecs_, elecs); 

  NrElec_->Fill( elecs->size() );
  for( std::vector<pat::Electron>::const_iterator elec=elecs->begin();
       elec!=elecs->end(); ++elec){
    // --------------------------------------------------
    // fill basic electron kinematics 
    // --------------------------------------------------
    ptElec_ ->Fill( elec->pt()    );
    enElec_ ->Fill( elec->energy());
    etaElec_->Fill( elec->eta()   );
    phiElec_->Fill( elec->phi()   );

    // --------------------------------------------------
    // ask for common tigger bits to be set
    // --------------------------------------------------
    edm::Handle<edm::TriggerResults> triggerBits;
    evt.getByLabel("TriggerResults",triggerBits);
    if (triggerBits.isValid()){
      for(unsigned iBit=0; iBit<triggerBits->size(); ++iBit){
	std::cout << "HLTTrigger bit " << iBit << " : " 
		  << triggerBits->at(iBit).accept() << std::endl;
      }
    }

    // --------------------------------------------------
    // get matched trigger primitive and fill best match 
    // --------------------------------------------------
    int trigIdx=-1;
    double minDR=-1.;

    std::cout << "size=" << elec->triggerMatches().size() << std::endl;
    const std::vector<pat::TriggerPrimitive> trig = elec->triggerMatches();
    for(unsigned idx = 0; idx<trig.size(); ++idx){
      std::cout << "found trigger match from HLT filter: " 
		<< trig[idx].filterName() << std::endl;
      double dR=deltaR(trig[idx].eta(), trig[idx].phi(), elec->eta(), elec->phi());
      if( minDR<0 || dR<minDR ){
	minDR=dR;
	trigIdx=idx;
      }
    }
    if(trigIdx>=0){
      trigMatchElec_->Fill((trig[trigIdx].pt()-elec->pt())/elec->pt());
    }
  }
}

void TopElecAnalyzer::beginJob(const edm::EventSetup&)
{
}

void TopElecAnalyzer::endJob()
{
}
  
