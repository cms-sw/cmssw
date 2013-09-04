
#include "SLHCUpgradeSimulations/L1CaloTrigger/plugins/CaloTriggerAnalyzer.h"
#include "Math/GenVector/VectorUtil.h"
#include <iostream>
#include <iomanip>


CaloTriggerAnalyzer::CaloTriggerAnalyzer(const edm::ParameterSet& iConfig):
  src_(iConfig.getParameter<edm::InputTag>("src")),
  ref_(iConfig.getParameter<edm::InputTag>("ref")),
  DR_(iConfig.getParameter<double>("deltaR")),
  threshold_(iConfig.getParameter<double>("threshold")),
  maxEta_(iConfig.getUntrackedParameter<double>("maxEta",2.5))
{
   //now do what ever initialization is needed

  edm::Service<TFileService> fs;

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
  dR = fs->make<TH1F>("dR","delta(R)",50,0,2);
}


CaloTriggerAnalyzer::~CaloTriggerAnalyzer()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
CaloTriggerAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  //  std::cout << src_ << std::endl;
  edm::Handle<edm::View<reco::Candidate> > ref;
  edm::Handle<edm::View<reco::Candidate> > src;

  bool gotRef = iEvent.getByLabel(ref_,ref);
  bool gotSrc = iEvent.getByLabel(src_,src);

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
	  ptDenom->Fill(i->pt());
	  etaDenom->Fill(i->eta());
	  
	  //	  printf("ref pt = %f  eta = %f phi = %f\n",i->pt(),i->eta(),i->phi());
	  
	  if(gotSrc)
	    {
	      bool matched=false;
	      math::XYZTLorentzVector highestV(0.0001,0.,0.,0.0002);
	      for(edm::View<reco::Candidate>::const_iterator j = src->begin(); j!= src->end();++j)
		{
		  dR->Fill(ROOT::Math::VectorUtil::DeltaR(i->p4(),j->p4()));
		  if(j->pt()>threshold_)
		    {
		      if(ROOT::Math::VectorUtil::DeltaR(i->p4(),j->p4())<DR_) {
			//	printf("matched pt = %f  eta = %f phi = %f\n",j->pt(),j->eta(),j->phi());
			
			if(j->pt()>highestV.pt())
			  highestV = j->p4();
			
			matched=true;
		      }
		      
		    }
		}
	    if(matched)
	      {
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
	      }
	    }
	}
    }
  }
}

DEFINE_FWK_MODULE(CaloTriggerAnalyzer);
