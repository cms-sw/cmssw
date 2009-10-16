#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/PatCandidates/interface/Flags.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "TopQuarkAnalysis/Examples/plugins/TopElecAnalyzer.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"

TopElecAnalyzer::TopElecAnalyzer(const edm::ParameterSet& cfg):
  elecs_(cfg.getParameter<edm::InputTag>("input"))
{
  edm::Service<TFileService> fs;
  
  nrElec_ = fs->make<TH1I>("NrElec",  "Nr_{Elec}",   10,  0 , 10 );
  ptElec_ = fs->make<TH1F>("ptElec",  "pt_{Elec}",  100,  0.,300.);
  enElec_ = fs->make<TH1F>("enElec",  "en_{Elec}",  100,  0.,300.);
  etaElec_= fs->make<TH1F>("etaElec", "eta_{Elec}", 100, -3.,  3.);
  phiElec_= fs->make<TH1F>("phiElec", "phi_{Elec}", 100, -4.,  4.);
  dptElec_= fs->make<TH1F>("dptElec", "dpt_{Elec}", 100, -2.,  2.);
  denElec_= fs->make<TH1F>("denElec", "den_{Elec}", 100, -2.,  2.);
  genElec_= fs->make<TH1F>("genElec", "gen_{Elec}", 100, -2.,  2.);
  trgElec_= fs->make<TH1F>("trgElec", "trg_{Elec}", 100, -1.,  1.);

  CountInSize_T=fs->make<TH2D>("NumDepoInIsoConeVsConeSize_T","NumDepoInIsoConeVsConeSize_T",30,0.04,0.34,50,0.,50.);
  CountInSize_E=fs->make<TH2D>("NumDepoInIsoConeVsConeSize_E","NumDepoInIsoConeVsConeSize_E",30,0.04,0.34,50,0.,50.);
  CountInSize_H=fs->make<TH2D>("NumDepoInIsoConeVsConeSize_H","NumDepoInIsoConeVsConeSize_H",30,0.04,0.34,50,0.,50.);

  DepoInSize_T =fs->make<TH2D>("AmountDepoInIsoConeVsConeSize_T","AmountDepoInIsoConeVsConeSize_T",30,0.04,0.34,50,0.,50.);
  DepoInSize_E =fs->make<TH2D>("AmountDepoInIsoConeVsConeSize_E","AmountDepoInIsoConeVsConeSize_E",30,0.04,0.34,50,0.,50.);
  DepoInSize_H =fs->make<TH2D>("AmountDepoInIsoConeVsConeSize_H","AmountDepoInIsoConeVsConeSize_H",30,0.04,0.34,50,0.,50.);

  Count_Threshold_T=fs->make<TH2D>("NumDepoInIsoConeVsConeSize_Threshold_T","NumDepoInIsoConeVsConeSize_Threshold_T",30,0.04,0.34,50,0.,50.);
  Count_Threshold_E=fs->make<TH2D>("NumDepoInIsoConeVsConeSize_Threshold_E","NumDepoInIsoConeVsConeSize_Threshold_E",30,0.04,0.34,50,0.,50.);
  Count_Threshold_H=fs->make<TH2D>("NumDepoInIsoConeVsConeSize_Threshold_H","NumDepoInIsoConeVsConeSize_Threshold_H",30,0.04,0.34,50,0.,50.);

  Depo_Threshold_T=fs->make<TH2D>("AmountDepoInIsoConeVsConeSize_Threshold_T","AmountDepoInIsoConeVsConeSize_Threshold_T",30,0.04,0.34,50,0.,50.);
  Depo_Threshold_E=fs->make<TH2D>("AmountDepoInIsoConeVsConeSize_Threshold_E","AmountDepoInIsoConeVsConeSize_Threshold_E",30,0.04,0.34,50,0.,50.);
  Depo_Threshold_H=fs->make<TH2D>("AmountDepoInIsoConeVsConeSize_Threshold_H","AmountDepoInIsoConeVsConeSize_Threshold_H",30,0.04,0.34,50,0.,50.);
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
    // FIXME BEGIN - remove obsolete trigger code
    //int trigIdx =-1 ;
    //double minDR=-1.;
    //const std::vector<pat::TriggerPrimitive> trig = elec->triggerMatches();
    //for(unsigned idx = 0; idx<trig.size(); ++idx){
    //  std::cout << "Trigger Match: " << trig[idx].filterName() << std::endl;
    //  double dR=deltaR(trig[idx].eta(), trig[idx].phi(), elec->eta(), elec->phi());
    //  if( minDR<0 || dR<minDR ){
    //  minDR=dR;
    //  trigIdx=idx;
    //  }
    //}
    //if(trigIdx>=0){
    //  trgElec_->Fill((trig[trigIdx].pt()-elec->pt())/elec->pt());
    //}
    // FIXME END
    
    // --------------------------------------------------
    // get ElectronId 
    // --------------------------------------------------
    const std::vector<pat::Electron::IdPair> electronIDs = elec->electronIDs();
    for(unsigned idx=0; idx<electronIDs.size(); ++idx){
      std::cout << ::std::setw( 25 ) << ::std::left << electronIDs[idx].first << ":" << electronIDs[idx].second << std::endl;
    }

    // --------------------------------------------------
    // get userFunction 
    // --------------------------------------------------
    //std::cout << std::endl;
    //if(elec->userDataObject("relIso")){
    //  std::cout << ::std::setw( 12 ) << ::std::left << elec->userDataObject("relIso") << std::endl;
    //}
    //else{
    //  std::cout << ::std::setw( 12 ) << ::std::left << "userData ValueMap is empty..." << std::endl;
    //}
    
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

    // --------------------------------------------------
    // Isolation in the tracker
    // --------------------------------------------------

    const reco::IsoDeposit *TID=0;
    TID = elec->trackIsoDeposit();
   
    //Isolation Cone coordinates
    //double IsoCone_eta=TID->eta();
    //double IsoCone_phi=TID->phi();
    
    
    //Veto Cone coordinates
    //double VetoCone_eta =TID->veto().vetoDir.eta();
    //double VetoCone_phi =TID->veto().vetoDir.phi();
    //float  VetoCone_size=TID->veto().dR;
    
   
    //get the candidate tag
    //float Cand_Tag_T =TID->candEnergy();
    
    //get the deposit & count within the isolation cone with the different cone size
    double radius=0.0;
    for(int i=0;i<6;i++)
      {

       radius+=0.05;
       CountInSize_T->Fill(radius,TID->depositAndCountWithin(radius).second);
       DepoInSize_T ->Fill(radius,TID->depositAndCountWithin(radius).first);

       //If the deposit should exceed some threshold
       double Threshold=0.3;
       Count_Threshold_T->Fill(radius,TID->depositAndCountWithin(radius,reco::IsoDeposit::Vetos(),Threshold).second);
       Depo_Threshold_T ->Fill(radius,TID->depositAndCountWithin(radius,reco::IsoDeposit::Vetos(),Threshold).first);
 
     }

    // --------------------------------------------------
    // Isolation in the ecal & hcal
    // --------------------------------------------------
    const reco::IsoDeposit *EID=0;
    const reco::IsoDeposit *HID=0;
    EID = elec->ecalIsoDeposit();
    HID = elec->hcalIsoDeposit();


    //get the candidate tag
    //float Cand_Tag_E =EID->candEnergy();
    //float Cand_Tag_H =HID->candEnergy();


    //get the deposit & count within the isolation cone with the different cone size
    for(int i=1;i<30;i++)
      {

       double radius=i/100;
       CountInSize_E->Fill(radius,EID->depositAndCountWithin(radius).second);
       DepoInSize_E ->Fill(radius,EID->depositAndCountWithin(radius).first);

       CountInSize_H->Fill(radius,HID->depositAndCountWithin(radius).second);
       DepoInSize_H ->Fill(radius,HID->depositAndCountWithin(radius).first);

       //If the deposit should exceed some threshold
       double Threshold=0.3;
       Count_Threshold_E->Fill(radius,EID->depositAndCountWithin(radius,reco::IsoDeposit::Vetos(),Threshold).second);
       Depo_Threshold_E ->Fill(radius,EID->depositAndCountWithin(radius,reco::IsoDeposit::Vetos(),Threshold).first);

       Count_Threshold_H->Fill(radius,HID->depositAndCountWithin(radius,reco::IsoDeposit::Vetos(),Threshold).second);
       Depo_Threshold_H ->Fill(radius,HID->depositAndCountWithin(radius,reco::IsoDeposit::Vetos(),Threshold).first);
       
      }

  }
}

void TopElecAnalyzer::beginJob(const edm::EventSetup&)
{
}

void TopElecAnalyzer::endJob()
{
}
  
