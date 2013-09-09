
#include "SLHCUpgradeSimulations/L1CaloTrigger/plugins/CaloTriggerAnalyzerOnDataTrees.h"
#include "Math/GenVector/VectorUtil.h"
#include <iostream>
#include <iomanip>


CaloTriggerAnalyzerOnDataTrees::CaloTriggerAnalyzerOnDataTrees(const edm::ParameterSet& iConfig):
  vertices_(iConfig.getParameter<edm::InputTag>("VertexCollection")),
  SLHCsrc_(iConfig.getParameter<edm::InputTag>("SLHCsrc")),
  LHCsrc_(iConfig.getParameter<edm::InputTag>("LHCsrc")),
  LHCisosrc_(iConfig.getParameter<edm::InputTag>("LHCisosrc")),
  electrons_(iConfig.getParameter<edm::InputTag>("electrons")),
  iso_(iConfig.getParameter<double>("iso")),
  DR_(iConfig.getParameter<double>("deltaR")),
  threshold_(iConfig.getParameter<double>("threshold"))
{
  //now do what ever initialization is needed

  edm::Service<TFileService> fs;

  RRTree = fs->make<TTree>("RRTree","Tree containing RAW RECO info");

  RRTree->Branch("RecoEpt",&RecoEpt);
  RRTree->Branch("RecoEeta",&RecoEeta);
  RRTree->Branch("RecoEphi",&RecoEphi);

  RRTree->Branch("LHCL1pt",&LHCL1pt);
  RRTree->Branch("LHCL1eta",&LHCL1eta);
  RRTree->Branch("LHCL1phi",&LHCL1phi);

  RRTree->Branch("SLHCL1pt",&SLHCL1pt);
  RRTree->Branch("SLHCL1eta",&SLHCL1eta);
  RRTree->Branch("SLHCL1phi",&SLHCL1phi);
  RRTree->Branch("Vertices",&numVertices);

  RECOpt       = fs->make<TH1F>( "RECOpt"   , "RECOpt",  20  ,  0. , 100. );
  LHCpt       = fs->make<TH1F>( "LHCpt"   , "LHCpt",  20  ,  0. , 100. );
  SLHCpt       = fs->make<TH1F>( "SLHCpt" , "SLHCpt", 20  ,  0. , 100. );
  pt       = fs->make<TH1F>( "pt"      , "pt", 20  ,  0. , 100. );
  highestPt= fs->make<TH1F>( "highestPt"      , "highestPt", 50  ,  0. , 100. );


  SLHChighestPt= fs->make<TH1F>( "SLHChighestPt"      , "SLHChighestPt", 50  ,  0. , 100. );
  SLHCsecondPt = fs->make<TH1F>( "SLHCsecondHighestPt", "SLHCsecondHighestPt", 50  ,  0. , 100. );
  LHChighestPt= fs->make<TH1F>( "LHChighestPt"      , "LHChighestPt", 50  ,  0. , 100. );
  LHCsecondPt = fs->make<TH1F>( "LHCsecondHighestPt", "LHCsecondHighestPt", 50  ,  0. , 100. );

  
}


CaloTriggerAnalyzerOnDataTrees::~CaloTriggerAnalyzerOnDataTrees()
{

  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
CaloTriggerAnalyzerOnDataTrees::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<edm::View<reco::Candidate> > LHCsrc;
  edm::Handle<edm::View<reco::Candidate> > SLHCsrc;
  edm::Handle<edm::View<reco::Candidate> > LHCisosrc;

  edm::Handle<reco::GsfElectronCollection> electrons;
  edm::Handle<reco::VertexCollection> vertices;
  iEvent.getByLabel(vertices_, vertices);
  
  bool gotLHCsrc = iEvent.getByLabel(LHCsrc_,LHCsrc);
  bool gotSLHCsrc = iEvent.getByLabel(SLHCsrc_,SLHCsrc);
  bool gotLHCisosrc = iEvent.getByLabel(LHCisosrc_,LHCisosrc);
  bool gotRecoE = iEvent.getByLabel(electrons_,electrons);

    
  //////Looking for Isolated////////
  
  if(gotRecoE) {
    highRecoPt=0;
    //    secondPtf=0;

    numVertices = vertices->size();
    printf("Vertices: %f \n",numVertices);
    for(unsigned int i=1; i<electrons->size();++i){
      RecoEpt = -50;      
      RecoEeta = -50;
      RecoEphi = -50;
      
      bool passID = false;
      
      if((electrons->at(i).dr04TkSumPt() + electrons->at(i).dr04EcalRecHitSumEt() + electrons->at(i).dr04HcalTowerSumEt())/(electrons->at(i).pt())<0.15)//
	if(electrons->at(i).isEB()||electrons->at(i).isEE()) 
	  if(fabs(electrons->at(i).sigmaIetaIeta())<0.025)  //sigmaEtaEta_[type]) 
	    if(fabs(electrons->at(i).deltaEtaSuperClusterTrackAtVtx())<0.02)  //deltaEta_[type]) 
	      if(fabs(electrons->at(i).deltaPhiSuperClusterTrackAtVtx())<0.1)//deltaPhi_[type]) 
		if(fabs(electrons->at(i).hcalOverEcal())<0.01)    //hoE_[type])
		  {printf("\n Electrons pass ID\n");
		    if((electrons->at(i).dr03TkSumPt()+electrons->at(i).dr03EcalRecHitSumEt()+electrons->at(i).dr03HcalDepth1TowerSumEt())/electrons->at(i).pt()<0.15)
		      {
			passID=true;
			printf("\n Electrons Isolated!\n");}
		  }
      
      
	    if(passID == true){
	      RECOpt->Fill(electrons->at(i).pt());
	      RecoEpt=electrons->at(i).pt();
	      
	      RecoEeta=electrons->at(i).eta();
	      RecoEphi=electrons->at(i).phi();
	      //printf("RecoPt=%f\n",RecoEpt);
	      //printf("RecoEta=%f\n",RecoEeta);
	      //printf("RecoPhi=%f\n",RecoEphi);
	      LHCL1pt=-20;
	      LHCL1eta=-20;
	      LHCL1phi=-20;
	      SLHCL1pt=-20;
	      SLHCL1eta=-20;
	      SLHCL1phi=-20;
	      
	      highPt=0;
	      highEta=0;
	      highPhi=0;
	      //////pass to module to search through SLHC here
	  
	      if(gotSLHCsrc) {
	
		secondPtf=0;
		for(edm::View<reco::Candidate>::const_iterator j = SLHCsrc->begin(); j!= SLHCsrc->end();++j)
		  {
		    if(ROOT::Math::VectorUtil::DeltaR(electrons->at(i).p4(),j->p4())<DR_) {
		      if (j->pt()>highPt){
			secondPtf=highPt;
			highPt = j->pt();
			SLHCL1pt=j->pt();
			SLHCL1eta=j->eta();
			SLHCL1phi=j->phi();

		      } 
		      else if (j->pt()>secondPtf){
			secondPtf=j->pt();
		      }
		    }
	
		    SLHCpt->Fill(j->pt());
		  }

	      }

	      LHCL1pt =0;	   
	      LHCL1eta = 0;
	      LHCL1phi = 0;

	      if(gotLHCisosrc||gotLHCsrc) {
		highPt = 0;
		highPhi= -30;
		highEta= -30;
		secondPtf=0;
		for(edm::View<reco::Candidate>::const_iterator l = LHCisosrc->begin(); l!= LHCisosrc->end();++l)
		  {
		    if(ROOT::Math::VectorUtil::DeltaR(electrons->at(i).p4(),l->p4())<DR_) {
		      if (l->pt()>highPt)
			{
			  secondPtf = highPt;
			  highPt=l->pt();
			  LHCL1pt  = l->pt();
			  LHCL1eta = l->eta();
			  LHCL1phi = l->phi();
			  //printf("LHCiso_highpt=%f\n",highPt);
			} 
		    }
		    //LHCpt->Fill(l->pt());
		    
		  }

		if(iso_==0){
		for(edm::View<reco::Candidate>::const_iterator j = LHCsrc->begin(); j!= LHCsrc->end();++j)
		  {
		    if(ROOT::Math::VectorUtil::DeltaR(electrons->at(i).p4(),j->p4())<DR_) {
		      if (j->pt()>highPt)
			{
			  secondPtf = highPt;
			  highPt = j->pt();
			  LHCL1pt  = j->pt();
			  LHCL1eta = j->eta();
			  LHCL1phi = j->phi();
			  // printf("LHCnonIsohighpt=%f\n",highPt);
			} 
		      else if (j->pt()>secondPtf){
			secondPtf=j->pt();
		      }
		    }
		    LHCpt->Fill(j->pt());
		  }
		}
      	      }//gotLHCisosrc


	      if(RecoEpt>0){
		//if(iso_==1){printf("Isolated Particles\n");}

		//printf("\n reco pt: %f \n reco eta: %f \n reco phi: %f \n LHC pt: %f \n LHC eta: %f \n LHC phi: %f \n SLHC pt: %f \n SLHC eta: %f \n SLHC phi: %f \n",RecoEpt,RecoEeta,RecoEphi,LHCL1pt,LHCL1eta,LHCL1phi,SLHCL1pt,SLHCL1eta,SLHCL1phi);
	      }
	      RRTree->Fill();	      
	    }//passID==true


	    //printf("\n Filling!!\n");
    }//electron Cands

  }
  
  
  else
	RECOpt->Fill(0.0);
  
}



void CaloTriggerAnalyzerOnDataTrees::matchSLHC(const reco::Candidate * recoCAND){
  ///get  reco particle and compare to SLHC L1 particles
  ///Find highest pt and best delta R matched
  // (To be completed)




}

DEFINE_FWK_MODULE(CaloTriggerAnalyzerOnDataTrees);
