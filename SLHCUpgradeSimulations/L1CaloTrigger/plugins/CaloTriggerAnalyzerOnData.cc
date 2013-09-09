
#include "SLHCUpgradeSimulations/L1CaloTrigger/plugins/CaloTriggerAnalyzerOnData.h"
#include "Math/GenVector/VectorUtil.h"
#include <iostream>
#include <iomanip>


CaloTriggerAnalyzerOnData::CaloTriggerAnalyzerOnData(const edm::ParameterSet& iConfig):
  SLHCsrc_(iConfig.getParameter<edm::InputTag>("SLHCsrc")),
  LHCsrc_(iConfig.getParameter<edm::InputTag>("LHCsrc")),
  LHCisosrc_(iConfig.getParameter<edm::InputTag>("LHCisosrc")),
  iso_(iConfig.getParameter<double>("iso"))
{
  //now do what ever initialization is needed

  edm::Service<TFileService> fs;

  LHCpt       = fs->make<TH1F>( "LHCpt"   , "LHCpt",  20  ,  0. , 100. );
  SLHCpt       = fs->make<TH1F>( "SLHCpt" , "SLHCpt", 20  ,  0. , 100. );
  pt       = fs->make<TH1F>( "pt"      , "pt", 20  ,  0. , 100. );
  highestPt= fs->make<TH1F>( "highestPt"      , "highestPt", 50  ,  0. , 100. );
  secondPt = fs->make<TH1F>( "secondHighestPt", "secondHighestPt", 50  ,  0. , 100. );

  SLHChighestPt= fs->make<TH1F>( "SLHChighestPt"      , "SLHChighestPt", 50  ,  0. , 100. );
  SLHCsecondPt = fs->make<TH1F>( "SLHCsecondHighestPt", "SLHCsecondHighestPt", 50  ,  0. , 100. );
  LHChighestPt= fs->make<TH1F>( "LHChighestPt"      , "LHChighestPt", 50  ,  0. , 100. );
  LHCsecondPt = fs->make<TH1F>( "LHCsecondHighestPt", "LHCsecondHighestPt", 50  ,  0. , 100. );

  
}


CaloTriggerAnalyzerOnData::~CaloTriggerAnalyzerOnData()
{

   // do anything here that needs to be done at destruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
CaloTriggerAnalyzerOnData::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  //  std::cout << src_ << std::endl;
  //  edm::Handle<edm::View<reco::Candidate> > ref;
  //edm::Handle<edm::View<reco::Candidate> > src;
  edm::Handle<edm::View<reco::Candidate> > LHCsrc;
  edm::Handle<edm::View<reco::Candidate> > SLHCsrc;
  edm::Handle<edm::View<reco::Candidate> > LHCisosrc;


  //bool gotRef = iEvent.getByLabel(ref_,ref);
  //Bool gotSrc = iEvent.getByLabel(src_,src);
  bool gotLHCsrc = iEvent.getByLabel(LHCsrc_,LHCsrc);
  bool gotSLHCsrc = iEvent.getByLabel(SLHCsrc_,SLHCsrc);
  bool gotLHCisosrc = iEvent.getByLabel(LHCisosrc_,LHCisosrc);


  if(iso_==1){
    if(gotLHCisosrc) {
      highPt=0;
      secondPtf=0;
      for(edm::View<reco::Candidate>::const_iterator i = LHCisosrc->begin(); i!= LHCisosrc->end();++i)
	{
	  LHCpt->Fill(i->pt());
	  if (i->pt()>highPt){
	    secondPtf=highPt;
	    highPt=i->pt();
	  } else if (i->pt()>secondPtf){
	    secondPtf=i->pt();
	  }
	}
    
      if(LHCisosrc->size()>0){
	LHChighestPt->Fill(highPt);
	printf("LHCsrc highpt= %f \n",highPt);
      }
	else
	LHChighestPt->Fill(0.0);

      if(LHCisosrc->size()>1)
	LHCsecondPt->Fill(secondPtf);
      else
	LHCsecondPt->Fill(0.0);
    }
  }


  if(iso_==0){
      highPt=0;
      secondPtf=0;

      if(gotLHCsrc) {
	for(edm::View<reco::Candidate>::const_iterator i = LHCsrc->begin(); i!= LHCsrc->end();++i)
	  {
	    LHCpt->Fill(i->pt());
	    if (i->pt()>highPt){
	      secondPtf=highPt;
	      highPt=i->pt();
	      //printf("LHCsrc highpt= %f \n",highPt);
	    } else if (i->pt()>secondPtf){
	      secondPtf=i->pt();
	    }
	  }
      }

      if(gotLHCisosrc) {
	//printf("Got Isolated LHC \n");
	for(edm::View<reco::Candidate>::const_iterator j = LHCisosrc->begin(); j!= LHCisosrc->end();++j)
	  {
	    //printf("LHCisosrc highpt= %f \n",highPt);
	    if (j->pt()>highPt){
	      secondPtf=highPt;
	      highPt=j->pt();

	    } else if (j->pt()>secondPtf){
	      secondPtf=j->pt();
	    }
	  }
      }
      
      LHChighestPt->Fill(highPt);
      //printf("LHCsrc highpt= %f \n",highPt);

      LHCsecondPt->Fill(secondPtf);
  }

  if(iso_ == 1 )
    printf("\n LHC ISO highPT:%f \n",highPt);
  else
    printf("\n LHC highPT:%f \n",highPt);
  
  if(gotSLHCsrc){
    highPt=0;
    secondPtf=0;
    for(edm::View<reco::Candidate>::const_iterator i = SLHCsrc->begin(); i!= SLHCsrc->end();++i)
      {
	SLHCpt->Fill(i->pt());
	if (i->pt()>highPt){
	  secondPtf=highPt;
	  highPt=i->pt();
	} else if (i->pt()>secondPtf){
	  secondPtf=i->pt();
	}
      }


    //if(highPt != highPtold){
    if(SLHCsrc->size()>0){
      SLHChighestPt->Fill(highPt);
    }
    else
      SLHChighestPt->Fill(0.0);


    if(SLHCsrc->size()>1)
      SLHCsecondPt->Fill(secondPtf);
    else
      SLHCsecondPt->Fill(0.0);
    
      if(iso_ == 1 )
	printf("\n SLHC ISO highPT:%f \n",highPt);
      else
	printf("\n SLHC highPT:%f \n",highPt);

  }
    
  


}

DEFINE_FWK_MODULE(CaloTriggerAnalyzerOnData);
