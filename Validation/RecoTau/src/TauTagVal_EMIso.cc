// -*- C++ -*-
//
// Package:    TauTagVal_EMIso
// Class:      TauTagVal_EMIso
// 
/**\class TauTagVal_EMIso TauTagVal_EMIso.cc RecoTauTag/ConeIsolation/test/TauTagVal_EMIso.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Simone Gennai
//         Created:  Wed Apr 12 11:12:49 CEST 2006
// $Id: TauTagVal_EMIso.cc,v 1.3 2007/03/05 18:23:34 gennai Exp $
//
//


// user include files
#include "Validation/RecoTau/interface/TauTagVal_EMIso.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"
#include "TLorentzVector.h"
#include "TH1D.h"
#include "TF1.h"
#include "TClonesArray.h"
#include <vector>


using namespace edm;
using namespace std;
using namespace reco;



TauTagVal_EMIso::TauTagVal_EMIso(const edm::ParameterSet& iConfig)
{
  nEvent = 0;
  jetTagSrc = iConfig.getParameter<InputTag>("JetTagProd");
  outPutFile = iConfig.getParameter<string>("OutPutFile");
 
  nEventsRiso.reserve(6);
  nEventsUsed.reserve(6);
  nEventsEnergy.reserve(6);
  nEventsEnergyUsed.reserve(6);
  for(int i=0;i<6;i++)
    {    
      nEventsRiso[i]=0;
      nEventsUsed[i]=0;
      nEventsEnergy[i]=0;
      nEventsEnergyUsed[i]=0;
    }

  nEventsUsed07.reserve(200);
  nEventsRiso07.reserve(200);

  nEventsUsed107.reserve(200);
  nEventsRiso107.reserve(200);

  nEventsUsed207.reserve(200);
  nEventsRiso207.reserve(200);

  nEventsUsed307.reserve(200);
  nEventsRiso307.reserve(200);

  for(int i=0;i<100;i++)
    {
      nEventsRiso07[i]=0.;
      nEventsUsed07[i]=0.;


      nEventsRiso107[i]=0.;
      nEventsUsed107[i]=0.;


      nEventsRiso207[i]=0.;
      nEventsUsed207[i]=0.;

      nEventsRiso307[i]=0.;
      nEventsUsed307[i]=0.;


    }


  ratio.reserve(100);
  ratioEta.reserve(100);
  etbin.reserve(100);
  nRuns=11;
  TString s="s";
  TString rationame="r";
  TString rationameEta="rEta";
  for(int i=0;i<20;i++)
    {
      rationame+=s;
      rationameEta+=s;
      hRatio=new TH1D(rationame,rationame,100,0,2);
      hRatioEta=new TH1D(rationameEta,rationameEta,100,0,2);
      ratio[i]=hRatio;
      ratioEta[i]=hRatioEta;
    }


  
  
DaqMonitorBEInterface* dbe = &*edm::Service<DaqMonitorBEInterface>();
  if(dbe) {
   
    dbe->setCurrentFolder("Efficiency"+jetTagSrc.label());
    effVsRiso07=dbe->book1D("EffVsRisoRsig07_130Et150","EffVsRisoRsig07_130Et150",50,0.,14.);
    effVsRiso107=dbe->book1D("EffVsRisoRsig07_80Et110","EffVsRisoRsig07_80Et110",50,0.,14.);
    effVsRiso207=dbe->book1D("EffVsRisoRsig07_50Et70","EffVsRisoRsig07_50Et70",50,0.,14.);
    effVsRiso307=dbe->book1D("EffVsRisoRsig07_30Et50","EffVsRisoRsig07_30Et50",50,0.,14.);    
}

    
  if (outPutFile.empty ()) {
    LogInfo("OutputInfo") << " TauJet histograms will NOT be saved";
  } else {
    LogInfo("OutputInfo") << " TauJethistograms will be saved to file:" << outPutFile;
  }
  
}

void TauTagVal_EMIso::beginJob(){ 

}

void TauTagVal_EMIso::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace reco;

  Handle<HepMCProduct> evt;
  // iEvent.getByLabel("VtxSmeared", evt);
  iEvent.getByLabel("source", evt);
  HepMC::GenEvent * myGenEvent = new  HepMC::GenEvent(*(evt->GetEvent()));
  int jj=-1;
  TClonesArray* TauJets=new TClonesArray("TLorentzVector");
  TLorentzVector TauMC(0.0,0.0,0.0,0.0);
  TLorentzVector TauJetMC(0.0,0.0,0.0,0.0);
  for ( HepMC::GenEvent::particle_iterator p = myGenEvent->particles_begin();
	p != myGenEvent->particles_end(); ++p ) {
     
    if ( abs((*p)->pdg_id()) !=0 && abs((*p)->pdg_id())<1000){

      bool FinalTau=true;
      TLorentzVector TauNet(0.0,0.0,0.0,0.);
      if(abs((*p)->pdg_id())==15)
	{
	  vector<HepMC::GenParticle*> TauDaught;
	  TauDaught=Daughters((*p));
	  for(vector<HepMC::GenParticle*>::iterator pit=TauDaught.begin();pit!=TauDaught.end();++pit)
	    {
	      if(abs((*pit)->pdg_id())==15) FinalTau=false;
	      if(abs((*pit)->pdg_id())==16) TauNet=TLorentzVector((*pit)->momentum().px(),(*pit)->momentum().py(),(*pit)->momentum().pz(),(*pit)->momentum().e());
	    }
	 
	  if(FinalTau)
	    {
	      TLorentzVector theTau((*p)->momentum().x(),(*p)->momentum().y(),(*p)->momentum().z(),(*p)->momentum().e());
	      TauJetMC=theTau-TauNet;
	      jj++;
	      new((*TauJets)[jj])TLorentzVector(TauJetMC);
	    }
	}
    }
  }

 


  Handle<EMIsolatedTauTagInfoCollection> tauTagInfoHandle;
  iEvent.getByLabel(jetTagSrc, tauTagInfoHandle);
  
  const EMIsolatedTauTagInfoCollection & tauTagInfo = *(tauTagInfoHandle.product());
  
  EMIsolatedTauTagInfoCollection::const_iterator i = tauTagInfo.begin();
  int it=0,it1=0;
  for (; i != tauTagInfo.end(); ++i) {

    //Added By Konstantinos A. Petridis
    TLorentzVector recTauJet(i->jet().px(),i->jet().py(),i->jet().pz(),i->jet().energy());
    bool mtchdTauJet=false;
    TLorentzVector* mcTauJetMtchd=0;
    for(int l=0;l<TauJets->GetEntriesFast();l++)
      {
	TLorentzVector* trTauJets=(TLorentzVector*)TauJets->At(l);
	if(dR(trTauJets,&recTauJet)<0.2)
	  {
	    mtchdTauJet=true;
	    mcTauJetMtchd=trTauJets;
	    }
      }
    
    if(mtchdTauJet)
      {

	for(int ii=0;ii<50;ii++){
	  double riso=0.0+ii*0.25;
	  double discriminator = i->discriminator(0.4,0.13,riso);
	  //cout <<"Discriminator "<<discriminator<<endl;
	  if(mcTauJetMtchd->Et()>130.0&&mcTauJetMtchd->Et()<150.0)
	    {
	      nEventsUsed07[ii]++;
	      if(discriminator){
		nEventsRiso07[ii]++;
	      }
	    }
	  if(mcTauJetMtchd->Et()>80.0&&mcTauJetMtchd->Et()<110.0)
	    {
	      nEventsUsed107[ii]++;
	      if(discriminator){
		nEventsRiso107[ii]++;
	      }
	    }
	  if(mcTauJetMtchd->Et()>50.0&&mcTauJetMtchd->Et()<70.0)
	    {
	      nEventsUsed207[ii]++;
	      if(discriminator){
		nEventsRiso207[ii]++;
	      }
	    }
	  if(mcTauJetMtchd->Et()>30.0&&mcTauJetMtchd->Et()<50.0)
	    {
	      nEventsUsed307[ii]++;
	      if(discriminator){
		nEventsRiso307[ii]++;
	      }
	    }
	}
      }
    it1++;
  }
  delete myGenEvent;
  delete TauJets;
}
  
void TauTagVal_EMIso::endJob(){
  int ibin;
  

  //Added By Konstantinos A. Petridis
  
  int ibin07=0,ibin107=0,ibin207=0,ibin307=0;
  
  for(int ii=0; ii<50; ii++){
    
    if(nEventsUsed07[ii] > 0){ibin07= ii+1;}
    if(nEventsUsed107[ii] > 0){ibin107= ii+1;}
    if(nEventsUsed207[ii] > 0){ibin207= ii+1;}
    if(nEventsUsed307[ii] > 0){ibin307= ii+1;}
  
    if(nEventsUsed07[ii])
      {
	float effRiso= nEventsRiso07[ii]/nEventsUsed07[ii];
	float effRisoErr=sqrt(effRiso*(1-effRiso)/nEventsUsed07[ii]);
	effVsRiso07->setBinContent(ibin07,effRiso);
	effVsRiso07->setBinError(ibin07,effRisoErr); 

      }

    if(nEventsUsed107[ii])
      {   
	float effRiso1= nEventsRiso107[ii]/nEventsUsed107[ii];
	float effRisoErr1=sqrt(effRiso1*(1-effRiso1)/nEventsUsed107[ii]);
	effVsRiso107->setBinContent(ibin107,effRiso1);
	effVsRiso107->setBinError(ibin107,effRisoErr1); 
      }

    if(nEventsUsed207[ii])
      {
	float effRiso2= nEventsRiso207[ii]/nEventsUsed207[ii];
	float effRisoErr2=sqrt(effRiso2*(1-effRiso2)/nEventsUsed207[ii]);
	effVsRiso207->setBinContent(ibin207,effRiso2);
	effVsRiso207->setBinError(ibin207,effRisoErr2); 
      }	

    if(nEventsUsed307[ii])
      {
	float effRiso3= nEventsRiso307[ii]/nEventsUsed307[ii];
	float effRisoErr3=sqrt(effRiso3*(1-effRiso3)/nEventsUsed307[ii]);
	effVsRiso307->setBinContent(ibin307,effRiso3);
	effVsRiso307->setBinError(ibin307,effRisoErr3); 
      }
  }

  double piEtmean[20];
  double piEtpar[100];
  double piEtmeanEr[20];
  double piEtamean[20];
  double piEtapar[100];
  double piEtameanEr[20];
  double etbinmean[20];
  double etbinerr[20];
  double etabinmean[20];
  double etabinerr[20];

  for(int i=0;i<nRuns;i++)
    {
      TF1 piEt("piEt","gaus");
      TF1 piEta("piEta","gaus");
      ratio[i]->Fit("piEt");
      ratioEta[i]->Fit("piEta");
      piEt.GetParameters(&piEtpar[0]);
      piEtmean[i]=piEtpar[1];
      piEtmeanEr[i]=piEt.GetParError(1);
      piEta.GetParameters(&piEtapar[0]);
      piEtamean[i]=piEtapar[1];
      piEtameanEr[i]=piEta.GetParError(1);
      etbinmean[i]=420/(nRuns*1.0)*i+420/(nRuns*2.0);
      etbinerr[i]=420/(nRuns*2.0);
      etabinmean[i]=2.2/(nRuns*1.0)*i+2.2/(nRuns*2.0);
      etabinerr[i]=2.2/(nRuns*2.0);
    }


  effVsRiso07->setAxisRange(0.4,1.0,2);
  effVsRiso107->setAxisRange(0.4,1.0,2);
  effVsRiso207->setAxisRange(0.4,1.0,2);
  effVsRiso307->setAxisRange(0.4,1.0,2);



  if (!outPutFile.empty() && &*edm::Service<DaqMonitorBEInterface>()) edm::Service<DaqMonitorBEInterface>()->save (outPutFile);
  
}
