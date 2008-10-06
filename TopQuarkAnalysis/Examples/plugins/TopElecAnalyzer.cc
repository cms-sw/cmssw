#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/PatCandidates/interface/Flags.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "TopQuarkAnalysis/Examples/plugins/TopElecAnalyzer.h"


TopElecAnalyzer::TopElecAnalyzer(const edm::ParameterSet& cfg):
  elecs_(cfg.getParameter<edm::InputTag>("input"))
{
  edm::Service<TFileService> fs;
  
  nrElec_ = fs->make<TH1I>("NrElec",  "Nr_{Elec}",   10,  0 , 10 );
  ptElec_ = fs->make<TH1F>("ptElec",  "pt_{Elec}",  100,  0.,300.);
  enElec_ = fs->make<TH1F>("enElec",  "en_{Elec}",  100,  0.,300.);
  etaElec_= fs->make<TH1F>("etaElec", "eta_{Elec}", 100, -3.,  3.);
  phiElec_= fs->make<TH1F>("phiElec", "phi_{Elec}", 100, -5.,  5.);
  dptElec_= fs->make<TH1F>("dptElec", "dpt_{Elec}", 100, -2.,  2.);
  denElec_= fs->make<TH1F>("denElec", "den_{Elec}", 100, -2.,  2.);
  genElec_= fs->make<TH1F>("genElec", "gen_{Elec}", 100, -2.,  2.);
  trgElec_= fs->make<TH1F>("trgElec", "trg_{Elec}", 100, -1.,  1.);
}

TopElecAnalyzer::~TopElecAnalyzer()
{
}

void
TopElecAnalyzer::analyze(const edm::Event& evt, const edm::EventSetup& setup)
{       
  edm::Handle<std::vector<pat::Electron> > elecs;
  evt.getByLabel(elecs_, elecs); 

  nrElec_->Fill( elecs->size() );
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
    // request a bunch of pat tags
    // --------------------------------------------------
    if(pat::Flags::test(*elec, pat::Flags::Isolation::Tracker)){
      std::cout << "Electron is Tracker Isolated" << std::endl;
    } 

    // --------------------------------------------------
    // check tigger bits
    // --------------------------------------------------
    // still needs to get a sensible implementation 
    // as soon as the trigger bits become available
    edm::Handle<edm::TriggerResults> triggerBits;
    evt.getByLabel("TriggerResults",triggerBits);

    unsigned bit = 0;
    if(triggerBits.isValid()){
      std::cout << "Trigger Bit [" << bit << "] = " << triggerBits->at(bit).accept() << std::endl;
    } 

    // --------------------------------------------------
    // get matched trigger primitives and fill best match
    // --------------------------------------------------
    int trigIdx =-1 ;
    double minDR=-1.;
    const std::vector<pat::TriggerPrimitive> trig = elec->triggerMatches();
    for(unsigned idx = 0; idx<trig.size(); ++idx){
      std::cout << "Trigger Match: " << trig[idx].filterName() << std::endl;
      double dR=deltaR(trig[idx].eta(), trig[idx].phi(), elec->eta(), elec->phi());
      if( minDR<0 || dR<minDR ){
	minDR=dR;
	trigIdx=idx;
      }
    }
    if(trigIdx>=0){
      trgElec_->Fill((trig[trigIdx].pt()-elec->pt())/elec->pt());
    }
    
    // --------------------------------------------------
    // get ElectronId 
    // --------------------------------------------------
    const std::vector<pat::Electron::IdPair> leptonIDs = elec->leptonIDs();
    for(unsigned idx=0; idx<leptonIDs.size(); ++idx){
      std::cout << ::std::setw( 12 ) << ::std::left << leptonIDs[idx].first << leptonIDs[idx].second << std::endl;
    }

    // --------------------------------------------------
    // get userFunction 
    // --------------------------------------------------
    std::cout << std::endl;
    if(elec->userDataObject("relIso")){
      std::cout << ::std::setw( 12 ) << ::std::left << elec->userDataObject("relIso") << std::endl;
    }
    else{
      std::cout << ::std::setw( 12 ) << ::std::left << "userData ValueMap is empty..." << std::endl;
    }
    
    // --------------------------------------------------
    // get embedded objects
    // --------------------------------------------------
    bool track=true, gsfTrack=true, superClus=true, genMatch=true;

    if(!elec->track()){
      track=false;
      std::cout << "TrackRef     : is not valid" << std::endl;
    }
    if(!elec->gsfTrack()){
      gsfTrack=false;
      std::cout << "gsfTrackRef  : is not valid" << std::endl;
    }
    if(!elec->superCluster()){
      superClus=false;
      std::cout << "superCluster : is not valid" << std::endl;
    }
    if(!elec->genLepton()){
      genMatch=false;
      std::cout << "genMatchRef  : is not valid" << std::endl;
    }

    if(gsfTrack && track    ){
      dptElec_->Fill( (elec->track()->pt() - elec->gsfTrack()->pt())/elec->gsfTrack()->pt() );
    }
    if(gsfTrack && superClus){
      denElec_->Fill( (elec->superCluster()->energy()- elec->gsfTrack()->pt())/elec->gsfTrack()->pt() );
    }
    //needs fix in PAT
    if(elec->genLepton()){
      genElec_->Fill( (elec->gsfTrack()->pt() - elec->genLepton()->pt())/elec->genLepton()->pt() );
    }
  }
}

void TopElecAnalyzer::beginJob(const edm::EventSetup&)
{
}

void TopElecAnalyzer::endJob()
{
}
  
