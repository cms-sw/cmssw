#include "SLHCUpgradeSimulations/L1CaloTrigger/plugins/CaloTriggerAnalyzer2.h"
#include "Math/GenVector/VectorUtil.h"
#include <iostream>
#include <iomanip>


CaloTriggerAnalyzer2::CaloTriggerAnalyzer2(const edm::ParameterSet& iConfig):
  ref_(iConfig.getParameter<edm::InputTag>("ref")),
  SLHCsrc_(iConfig.getParameter<edm::InputTag>("SLHCsrc")),
  LHCsrc_(iConfig.getParameter<edm::InputTag>("LHCsrc")),
  DR_(iConfig.getParameter<double>("deltaR")),
  threshold_(iConfig.getParameter<double>("threshold")),
  maxEta_(iConfig.getUntrackedParameter<double>("maxEta",2.5))
{
   //now do what ever initialization is needed

  edm::Service<TFileService> fs;

  EventTree = fs->make<TTree>("EventTree","Tree containing event-by-event info");
  CandTree = fs->make<TTree>("CandTree","Tree containing candidate info");

  CandTree->Branch("gPt",&gPt);
  CandTree->Branch("gEta",&gEta);
  CandTree->Branch("gPhi",&gPhi);
 
  CandTree->Branch("SLHCpassSingleThresh",&SLHCpassSingleThresh);
  CandTree->Branch("SLHCL1Pt",&SLHCL1Pt);
  CandTree->Branch("SLHCL1Eta",&SLHCL1Eta);
  CandTree->Branch("SLHCL1Phi",&SLHCL1Phi);
  CandTree->Branch("SLHCdR",&SLHCdR);
  CandTree->Branch("SLHCdPt",&SLHCdPt);
  CandTree->Branch("SLHCdEta",&SLHCdEta);
  CandTree->Branch("SLHCdPhi",&SLHCdPhi);
  CandTree->Branch("SLHCRPt",&SLHCRPt);
  
  CandTree->Branch("LHCpassSingleThresh",&LHCpassSingleThresh);
  CandTree->Branch("LHCL1Pt",&LHCL1Pt);
  CandTree->Branch("LHCL1Eta",&LHCL1Eta);
  CandTree->Branch("LHCL1Phi",&LHCL1Phi);
  CandTree->Branch("LHCdR",&LHCdR);
  CandTree->Branch("LHCdPt",&LHCdPt);
  CandTree->Branch("LHCdEta",&LHCdEta);
  CandTree->Branch("LHCdPhi",&LHCdPhi);
  CandTree->Branch("LHCRPt",&LHCRPt);
    
  EventTree->Branch("ghighPt",&ghighPt);
  EventTree->Branch("gsecondPt",&gsecondPt);
  EventTree->Branch("ghighMPt",&ghighMPt);
  EventTree->Branch("SLHChighPt",&SLHChighPt);
  EventTree->Branch("SLHCsecondPt",&SLHCsecondPt);
  EventTree->Branch("LHChighPt",&LHChighPt);
  EventTree->Branch("LHCsecondPt",&LHCsecondPt);
  EventTree->Branch("nEvents",&nEvents);
  EventTree->Branch("SLHCsingleTrigger",&SLHCsingleTrigger);
  EventTree->Branch("SLHCdoubleTrigger",&SLHCdoubleTrigger);
  EventTree->Branch("LHCsingleTrigger",&LHCsingleTrigger);
  EventTree->Branch("LHCdoubleTrigger",&LHCdoubleTrigger);

  ptNum    = fs->make<TH1F>( "ptNum"   , "ptNum"   , 20  ,  0., 100. );
  ptDenom  = fs->make<TH1F>( "ptDenom" , "ptDenom" , 20  ,  0., 100. );
  etaNum   = fs->make<TH1F>( "etaNum"  , "etaNum"  , 20  ,  -2.5, 2.5 );
  etaDenom = fs->make<TH1F>( "etaDenom", "etaDenom", 20  ,  -2.5, 2.5 );
  pt       = fs->make<TH1F>( "pt"      , "pt", 20  ,  0. , 100. );
  highestPt= fs->make<TH1F>( "highestPt"      , "highestPt", 50  ,  0. , 100. );
  secondPt = fs->make<TH1F>( "secondHighestPt", "secondHighestPt", 50  ,  0. , 100. );
  highestPtGen=fs->make<TH1F>("highestPtGen","highestPtGen",100, 0., 200.);
  secondPtGen=fs->make<TH1F>("secondPtGen","secondPtGen",100, 0., 200.);
  dPt      = fs->make<TH1F>( "dPt"      , "dPt", 50  , -1  , 1 );
  dEta     = fs->make<TH1F>( "dEta"      , "dEta", 50  , -0.5  , 0.5 );
  dPhi     = fs->make<TH1F>( "dPhi"      , "dPhi", 50  , -0.5  , 0.5 );
  RPt = fs->make<TH1F>("RPt", "Pt_Ratio", 10, 0, 2.);
  RPtEta = fs->make<TProfile>("RPtEta","Pt_Ratio as fcn of abs(eta)",13,0.0,2.6,0,2);
  RPtEtaFull = fs->make<TProfile>("RPtEtaFull","Pt_Ratio as fcn of eta",26,-2.6,2.6,0,2);
  absEta = fs->make<TH1F>("abseta","abs(eta)",13,0.,2.6); 
  SLHCalldR = fs->make<TH1F>("SLHCalldR","delta(R) (SLHC)",50,0,2);
  LHCalldR = fs->make<TH1F>("LHCalldR","delta(R) (LHC)",50,0,2);

}


CaloTriggerAnalyzer2::~CaloTriggerAnalyzer2()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
CaloTriggerAnalyzer2::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  //  std::cout << src_ << std::endl;
  edm::Handle<edm::View<reco::Candidate> > SLHCsrc;
  edm::Handle<edm::View<reco::Candidate> > LHCsrc;
  edm::Handle<edm::View<reco::Candidate> > ref;
  edm::Handle<edm::View<reco::Candidate> > src;

  bool gotSLHCsrc = iEvent.getByLabel(SLHCsrc_,SLHCsrc);
  bool gotLHCsrc = iEvent.getByLabel(LHCsrc_,LHCsrc);
  bool gotRef = iEvent.getByLabel(ref_,ref);
  bool gotSrc =iEvent.getByLabel(SLHCsrc_,src);

  //clear variables...
  clearVectors();

  // const reco::Candidate * L1object;

  if(gotSrc) {
    highPt=0;
    secondPtf=0;
    for(edm::View<reco::Candidate>::const_iterator i = src->begin(); i!= src->end();++i)
      {
	pt->Fill(i->pt());
	if (i->pt()>highPt){
	  secondPtf=highPt;
	  highPt=i->pt();
	} else if (i->pt()>secondPtf){
	  secondPtf=i->pt();
	}
      }

    if(src->size()>0)
      highestPt->Fill(highPt);
    else
      highestPt->Fill(0.0);


    if(src->size()>1)
      secondPt->Fill(secondPtf);
    else
      secondPt->Fill(0.0);
  }

  if(gotRef) {
    //get highest Pt gen object--loop over all to make sure it's the highest!
    highestGenPt=-1.0;
    secondGenPt=-1.0;
    ghighMPt=0;
    for(edm::View<reco::Candidate>::const_iterator i = ref->begin(); i!= ref->end();++i){

      if (i->pt()>highestGenPt){
	highestGenPt=i->pt();
	highestPtGen->Fill(i->pt());
      } else if (i->pt()>secondGenPt){
	secondGenPt=i->pt();
	secondPtGen->Fill(i->pt());
      }
      if(fabs(i->eta())<maxEta_&&i->pt()>threshold_/2.)
	{
	  gPt.push_back(i->pt());
	  gEta.push_back(i->eta());
	  gPhi.push_back(i->phi());

	  ptDenom->Fill(i->pt());
	  etaDenom->Fill(i->eta());
	  
	  //printf("ref pt = %f  eta = %f phi = %f\n",i->pt(),i->eta(),i->phi());
	  
	  if(gotSLHCsrc) {
	    bool matched=false;
	    bool halfmatched=false;
	    double mindR=900;
	    L1object=NULL;
	    L1closest=NULL;
	    math::XYZTLorentzVector highestV(0.0001,0.,0.,0.0002);
	    for(edm::View<reco::Candidate>::const_iterator j = SLHCsrc->begin(); j!= SLHCsrc->end();++j)
	      {
		SLHCalldR->Fill(ROOT::Math::VectorUtil::DeltaR(i->p4(),j->p4()));
		
		if (ROOT::Math::VectorUtil::DeltaR(i->p4(),j->p4())<mindR){
		  mindR=ROOT::Math::VectorUtil::DeltaR(i->p4(),j->p4());
		  L1closest=&(*j);
		}

		if(ROOT::Math::VectorUtil::DeltaR(i->p4(),j->p4())<DR_) {
		  if(j->pt()>threshold_/2){
		    if(j->pt()>threshold_){
		      //printf("matched pt = %f  eta = %f phi = %f\n",j->pt(),j->eta(),j->phi());
		      matched=true;
		    }
		    halfmatched=true;
		  }
		  if(j->pt()>highestV.pt()){
		    highestV = j->p4();
		    L1object=&(*j);
		  }
		}
	      } //end for loop over L1 objects
	    
	    //Fill L1 object tree
	    fillSLHCBranches(L1object, &(*i));

	    if(halfmatched) {
	      SLHCpassDoubleThresh.push_back(true);
	    } else{
	      SLHCpassDoubleThresh.push_back(false);
	    }
	    if(matched) {
	      SLHCpassSingleThresh.push_back(true);

	      if (i->pt()>ghighMPt)
		ghighMPt=i->pt();
	      
	      RPt->Fill(i->pt()/highestV.pt());
	      RPtEtaFull->Fill(highestV.eta(), i->pt()/highestV.pt());
	      RPtEta->Fill(fabs(highestV.eta()), i->pt()/highestV.pt());
	      
	      //		printf("matched abs(eta) = %f \n", fabs(highestV.eta()) );
	      absEta->Fill(fabs(highestV.eta()));
	      dPt->Fill((highestV.pt()-i->pt())/i->pt());
	      dEta->Fill(highestV.eta()-i->eta());
	      dPhi->Fill(highestV.phi()-i->phi());
	      
	      ptNum->Fill(i->pt());
	      etaNum->Fill(i->eta());
	    } else {
	      SLHCpassSingleThresh.push_back(false);
	    }
	  } //end (if gotSLHCsrc)

          if(gotLHCsrc) {
            bool matched=false;
            bool halfmatched=false;
            double mindR=900;
	    L1object=NULL;
	    printf("LHC src is %i in size \n",LHCsrc->size());
	    math::XYZTLorentzVector highestV(0.0001,0.,0.,0.0002);
            for(edm::View<reco::Candidate>::const_iterator j = LHCsrc->begin(); j!= LHCsrc->end();++j)
              {
                LHCalldR->Fill(ROOT::Math::VectorUtil::DeltaR(i->p4(),j->p4()));

                if (ROOT::Math::VectorUtil::DeltaR(i->p4(),j->p4())<mindR){
                  mindR=ROOT::Math::VectorUtil::DeltaR(i->p4(),j->p4());
		  L1closest=&(*j);
		}
		
		if(ROOT::Math::VectorUtil::DeltaR(i->p4(),j->p4())<DR_) {
		  if(j->pt()>threshold_/2){
		    if(j->pt()>threshold_){
                      //printf("matched pt = %f  eta = %f phi = %f\n",j->pt(),j->eta(),j->phi());
                      matched=true;
                    }
                    halfmatched=true;
                  }
		  if(j->pt()>highestV.pt()){
		    highestV = j->p4();
		    L1object=&(*j); //Match to highest Pt object in cone.  If none is found, the written object is the closest in dR.
		  }
                }
              } //end for loop over L1 objects                                                                                                                

            //Fill L1 object tree   
	    fillLHCBranches(L1object, &(*i));
	    
	    if(halfmatched) {
              LHCpassDoubleThresh.push_back(true);
	    } else{
              LHCpassDoubleThresh.push_back(false);
            }
          
	    if(matched) {
              LHCpassSingleThresh.push_back(true);
	      RPt->Fill(i->pt()/highestV.pt());
              RPtEtaFull->Fill(highestV.eta(), i->pt()/highestV.pt());
              RPtEta->Fill(fabs(highestV.eta()), i->pt()/highestV.pt());

              //                printf("matched abs(eta) = %f \n", fabs(highestV.eta()) );                                                                    
              absEta->Fill(fabs(highestV.eta()));
              dPt->Fill((highestV.pt()-i->pt())/i->pt());
              dEta->Fill(highestV.eta()-i->eta());
              dPhi->Fill(highestV.phi()-i->phi());

              ptNum->Fill(i->pt());
              etaNum->Fill(i->eta());
            } else {
              LHCpassSingleThresh.push_back(false);
            }
          } //end (if gotLHCsrc) 
	  
	} //end (if GEN in acceptance)
    } //end GEN loop 
  }  //end (if gotRef)
  CandTree->Fill();
  
  //-------------------Event-by-event work-----------------------
  nEvents=0; //used for filling number of events.

  if(gotRef) {
    ghighPt=0.0;
    gsecondPt=0.0;
    for(edm::View<reco::Candidate>::const_iterator i = ref->begin(); i!= ref->end();++i)
      {
        if (i->pt()>ghighPt && fabs(i->eta())<maxEta_){
          gsecondPt=ghighPt;
          ghighPt=i->pt();
        } else if (i->pt()>gsecondPt && fabs(i->eta())<maxEta_){
          gsecondPt=i->pt();
        }
      }
  }

  if(gotSLHCsrc) {
    SLHChighPt=0.0;
    SLHCsecondPt=0.0;
    for(edm::View<reco::Candidate>::const_iterator i = SLHCsrc->begin(); i!= SLHCsrc->end();++i)
      {
        if (i->pt()>SLHChighPt){
          SLHCsecondPt=SLHChighPt;
          SLHChighPt=i->pt();
        } else if (i->pt()>SLHCsecondPt){
	  SLHCsecondPt=i->pt();
        }
      }
  }

  if(gotLHCsrc) {
    LHChighPt=0.0;
    LHCsecondPt=0.0;
    for(edm::View<reco::Candidate>::const_iterator i = LHCsrc->begin(); i!= LHCsrc->end();++i)
      {
        if (i->pt()>LHChighPt){
          LHCsecondPt=LHChighPt;
          LHChighPt=i->pt();
        } else if (i->pt()>LHCsecondPt){
          LHCsecondPt=i->pt();
        }
      }
  }

  SLHCsingleTrigger=false;
  for(unsigned int i=0; i<SLHCpassSingleThresh.size(); i++){
    if(SLHCpassSingleThresh.at(i))
      SLHCsingleTrigger=true;
  }
  SLHCdoubleTrigger=false;
  unsigned int nSLHCdouble=0;
  for(unsigned int i=0; i<SLHCpassDoubleThresh.size(); i++){
    if(SLHCpassDoubleThresh.at(i))
      nSLHCdouble++;
  }
  if(nSLHCdouble>=2)
    SLHCdoubleTrigger=true;

  LHCsingleTrigger=false;
  for(unsigned int i=0; i<LHCpassSingleThresh.size(); i++){
    if(LHCpassSingleThresh.at(i))
      LHCsingleTrigger=true;
  }
  LHCdoubleTrigger=false;
  unsigned int nLHCdouble=0;
  for(unsigned int i=0; i<LHCpassDoubleThresh.size(); i++){
    if(LHCpassDoubleThresh.at(i))
      nLHCdouble++;
  }
  if(nLHCdouble>=2)
    LHCdoubleTrigger=true;

  // printf("%s \n", (SLHCsingleTrigger && ghighPt>37.0)?"true":"false");

  EventTree->Fill();
}

void CaloTriggerAnalyzer2::clearVectors() {
  gPt.clear();
  gEta.clear();
  gPhi.clear();

  SLHCL1Pt.clear();
  SLHCL1Eta.clear();
  SLHCL1Phi.clear();
  SLHCdR.clear();
  SLHCdPt.clear();
  SLHCdEta.clear();
  SLHCdPhi.clear();
  SLHCRPt.clear();
  SLHCpassSingleThresh.clear();
  SLHCpassDoubleThresh.clear();
 
  LHCL1Pt.clear();
  LHCL1Eta.clear();
  LHCL1Phi.clear();
  LHCdR.clear();
  LHCdPt.clear();
  LHCdEta.clear();
  LHCdPhi.clear();
  LHCRPt.clear();
  LHCpassSingleThresh.clear();
  LHCpassDoubleThresh.clear();
}

void CaloTriggerAnalyzer2::fillSLHCBranches(const reco::Candidate * cand,  const reco::Candidate * genParticle){
  if(cand==NULL){
    //printf("No match found for SLHC!\n");
    SLHCL1Pt.push_back(-1);
    SLHCL1Eta.push_back(-137);
    SLHCL1Phi.push_back(-137);
    SLHCdR.push_back(-137);
    SLHCdPt.push_back(-137);
    SLHCdEta.push_back(-137);
    SLHCdPhi.push_back(-137);
    SLHCRPt.push_back(-137);
  } else {
    SLHCL1Pt.push_back(cand->pt());
    SLHCL1Eta.push_back(cand->eta());
    SLHCL1Phi.push_back(cand->phi());
    SLHCdR.push_back(ROOT::Math::VectorUtil::DeltaR(cand->p4(),genParticle->p4()));
    SLHCdPt.push_back((cand->pt()-genParticle->pt())/genParticle->pt());
    SLHCdEta.push_back(cand->eta()-genParticle->eta());
    SLHCdPhi.push_back(cand->phi()-genParticle->phi());
    SLHCRPt.push_back(genParticle->pt()/cand->pt());
  }
}

void CaloTriggerAnalyzer2::fillLHCBranches(const reco::Candidate * cand,  const reco::Candidate * genParticle){
  if(cand==NULL){
    //    printf("No match found for LHC!\n");
    LHCL1Pt.push_back(-1);
    LHCL1Eta.push_back(-137);
    LHCL1Phi.push_back(-137);
    LHCdR.push_back(-137);
    LHCdPt.push_back(-137);
    LHCdEta.push_back(-137);
    LHCdPhi.push_back(-137);
    LHCRPt.push_back(-137);
  } else{
    printf("LHC L1Pt=%f",cand->pt());
    LHCL1Pt.push_back(cand->pt());
    LHCL1Eta.push_back(cand->eta());
    LHCL1Phi.push_back(cand->phi());
    LHCdR.push_back(ROOT::Math::VectorUtil::DeltaR(cand->p4(),genParticle->p4()));
    LHCdPt.push_back((cand->pt()-genParticle->pt())/genParticle->pt());
    LHCdEta.push_back(cand->eta()-genParticle->eta());
    LHCdPhi.push_back(cand->phi()-genParticle->phi());
    LHCRPt.push_back(genParticle->pt()/cand->pt());
  }
}

DEFINE_FWK_MODULE(CaloTriggerAnalyzer2);
