// -*- C++ -*-
//
// Package:    TauTagVal
// Class:      TauTagVal
// 
/**\class TauTagVal TauTagVal.cc RecoTauTag/ConeIsolation/test/TauTagVal.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Simone Gennai
//         Created:  Wed Apr 12 11:12:49 CEST 2006
// $Id: TauTagVal.cc,v 1.7 2007/05/10 14:27:44 gennai Exp $
//
//


// user include files
#include "Validation/RecoTau/interface/TauTagVal.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"
#include "TF1.h"
#include "TClonesArray.h"



using namespace edm;
using namespace std;
using namespace reco;



TauTagVal::TauTagVal(const edm::ParameterSet& iConfig)
{
  nEvent = 0;
  jetTagSrc = iConfig.getParameter<InputTag>("JetTagProd");
  outPutFile = iConfig.getParameter<string>("OutPutFile");
  rSig = iConfig.getParameter<double>("SignalCone");
  rMatch = iConfig.getParameter<double>("MatchingCone");
  rIso = iConfig.getParameter<double>("IsolationCone");
  ptLeadTk = iConfig.getParameter<double>("MinimumTransverseMomentumLeadingTrack");

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

  nEventsUsed07.reserve(20);
  nEventsRiso07.reserve(20);
  nEventsUsed04.reserve(20);
  nEventsRiso04.reserve(20);

  nEventsUsed107.reserve(20);
  nEventsRiso107.reserve(20);
  nEventsUsed104.reserve(20);
  nEventsRiso104.reserve(20);

  nEventsUsed207.reserve(20);
  nEventsRiso207.reserve(20);
  nEventsUsed204.reserve(20);
  nEventsRiso204.reserve(20);

  nEventsUsed307.reserve(20);
  nEventsRiso307.reserve(20);
  nEventsUsed304.reserve(20);
  nEventsRiso304.reserve(20);

  for(int i=0;i<20;i++)
    {
      nEventsRiso07[i]=0.;
      nEventsUsed07[i]=0.;
      nEventsRiso04[i]=0.;
      nEventsUsed04[i]=0.;

      nEventsRiso107[i]=0.;
      nEventsUsed107[i]=0.;
      nEventsRiso104[i]=0.;
      nEventsUsed104[i]=0.;

      nEventsRiso207[i]=0.;
      nEventsUsed207[i]=0.;
      nEventsRiso204[i]=0.;
      nEventsUsed204[i]=0.;

      nEventsRiso307[i]=0.;
      nEventsUsed307[i]=0.;
      nEventsRiso304[i]=0.;
      nEventsUsed304[i]=0.;


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
    dbe->setCurrentFolder("TauJetTask_" + jetTagSrc.label());    
    ptLeadingTrack = dbe->book1D("PtLeadTk", "Pt LeadTk", 30, 0., 300.);
    ptJet  = dbe->book1D("PtJet", "Pt Jet", 30, 0., 300.);
    nSignalTracks = dbe->book1D("NSigTks", "NSigTks", 10, 0., 10.);
    nSignalTracksAfterIsolation = dbe->book1D("NSigTksAI", "NSigTksAI", 10, 0., 10.);
    nAssociatedTracks = dbe->book1D("NAssTks", "NAssTks", 10, 0., 10.);
    nSelectedTracks = dbe->book1D("NSelTks", "NSelTks", 10, 0., 10.);

    effVsRiso = dbe->book1D("EffIsol","EffIsol",6,0.2,0.5);
    EventseffVsRiso = dbe->book1D("EventsIsol","EvEffIsol",6,0.2,0.5);
    EventsToteffVsRiso = dbe->book1D("EventsTotIsol","EvTotEffIsol",6,0.2,0.5);
    effVsEt = dbe->book1D("EffVsEtJet","EffVsEtJet",6,0.,300.);
    EventseffVsEt = dbe->book1D("EventsEffVsEtJet","EvEffVsEtJet",6,0.,300.);
    EventsToteffVsEt = dbe->book1D("EventsTotEffVsEtJet","EvTotEffVsEtJet",6,0.,300.);
    effFindLeadTk =dbe->book1D("EffLeadTk","EffLeadTk",2,0.,2.);
   
    deltaRLeadTk_Jet = dbe->book1D("DeltaR_LT_Jet","DeltaR",20,0.,0.2);
    hDRRecLdgTrTauJet=dbe->book1D("DeltaRJetLtr40<Et<60","DeltaRJetLtr40<Et<60",16,0,0.16);
    hDRRecLdgTrTauJet1=dbe->book1D("DeltaRJetLtr200<Et<250","DeltaRJetLtr200<Et<250",16,0,0.16);;
    
    dbe->setCurrentFolder("IsolationConeEff"+jetTagSrc.label());
    effVsRiso07=dbe->book1D("EffVsRisoRsig07_130Et150","EffVsRisoRsig07_130Et150",20,0,1);
    effVsRiso107=dbe->book1D("EffVsRisoRsig07_80Et110","EffVsRisoRsig07_80Et110",20,0,1);
    effVsRiso207=dbe->book1D("EffVsRisoRsig07_50Et70","EffVsRisoRsig07_50Et70",20,0,1);
    effVsRiso307=dbe->book1D("EffVsRisoRsig07_30Et50","EffVsRisoRsig07_30Et50",20,0,1);
    effVsRiso04=dbe->book1D("EffVsRisoRsig04_130Et150","EffVsRisoRsig04_130Et150",20,0,1);
    effVsRiso104=dbe->book1D("EffVsRisoRsig04_80Et110","EffVsRisoRsig04_80Et110",20,0,1);
    effVsRiso204=dbe->book1D("EffVsRisoRsig04_50Et70","EffVsRisoRsig04_50Et70",20,0,1);
    effVsRiso304=dbe->book1D("EffVsRisoRsig04_30Et50","EffVsRisoRsig04_30Et50",20,0,1);
    
    dbe->setCurrentFolder("TauJetResponsePlots"+jetTagSrc.label());
    hEtmean=dbe->book1D("EtmeanResp","EtmeanResp",nRuns,0,420);
    hEtamean=dbe->book1D("EtameanResp","EtameanResp",nRuns,0,2.2);
    hTauJets=dbe->book1D("hTauJets","hTauJets",200,0,450);

}

    
  if (outPutFile.empty ()) {
    LogInfo("OutputInfo") << " TauJet histograms will NOT be saved";
  } else {
    LogInfo("OutputInfo") << " TauJethistograms will be saved to file:" << outPutFile;
  }
  
}

void TauTagVal::beginJob(){ 

}

void TauTagVal::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace reco;

  Handle<HepMCProduct> evt;
  //iEvent.getByLabel("VtxSmeared", evt);
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
	      if(abs((*pit)->pdg_id())==15)FinalTau=false;
	      if(abs((*pit)->pdg_id())==16)TauNet=TLorentzVector((*pit)->momentum().px(),(*pit)->momentum().py(),(*pit)->momentum().pz(),(*pit)->momentum().e());
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

 
  for(int j=0;j<TauJets->GetEntriesFast();j++)
    {
      TLorentzVector* theTauJet=(TLorentzVector*)TauJets->At(j);
      hTauJets->Fill(theTauJet->Et());
    }


  Handle<IsolatedTauTagInfoCollection> tauTagInfoHandle;
  iEvent.getByLabel(jetTagSrc, tauTagInfoHandle);
  
  const IsolatedTauTagInfoCollection & tauTagInfo = *(tauTagInfoHandle.product());
  
  IsolatedTauTagInfoCollection::const_iterator i = tauTagInfo.begin();
  int it=0,it1=0;
  for (; i != tauTagInfo.end(); ++i) {

    //Take only the first jet waiting for a jet selector
    if(it == 0) {
      for(int ii=0;ii<6;ii++)
	{
	  nEventsUsed[ii]++;
	  float Riso = ii*0.05 + 0.2;
	  float Etmin = ii*50.;
	  float Etmax  = Etmin+50;
	  float Rmatch = rMatch;
	  float Rsig = rSig;
	  float pT_LT = ptLeadTk;
	  float pT_min =1.;
	  if( i->discriminator(Rmatch,Rsig,Riso,pT_LT,pT_min) > 0) {
	    nEventsRiso[ii]++;
	  }
	  if(i->jet()->pt() > Etmin && i->jet()->pt()<Etmax)
	    {
	      nEventsEnergyUsed[ii]++;
	      if( i->discriminator(Rmatch,Rsig,rIso,pT_LT,pT_min) > 0) nEventsEnergy[ii]++;
	    }
	  if(!(i->leadingSignalTrack(rMatch, ptLeadTk)))
	    {
	      effFindLeadTk->Fill(0.);
	    }else{
	      effFindLeadTk->Fill(1.);
	    }
	  const TrackRef leadTkTmp= (i->leadingSignalTrack(0.5, 1.));
	  if(!leadTkTmp){
	  }else{
	    math::XYZVector momentum = (*leadTkTmp).momentum();
	    math::XYZVector jetMomentum(i->jet()->px(), i->jet()->py(), i->jet()->pz());
	    float deltaR = ROOT::Math::VectorUtil::DeltaR(jetMomentum, momentum);
	    deltaRLeadTk_Jet->Fill(deltaR);
	  }
	  
	  const TrackRef leadTk= (i->leadingSignalTrack(rMatch, 1.));
	  if(!leadTk){
	    LogInfo("LeadingTrack") << " No LeadingTrack";
	  }else{
	    ptLeadingTrack->Fill((*leadTk).pt());
	    ptJet->Fill((i->jet())->pt());
	    math::XYZVector momentum = (*leadTk).momentum();
	    float nsigtks = (i->tracksInCone(momentum, rSig,  1.)).size();
	    nSignalTracks->Fill(nsigtks);
	    if(i->discriminator(rMatch,rSig,rIso,ptLeadTk,1.) == 1)
	      nSignalTracksAfterIsolation->Fill(nsigtks);
	  }
	  float allTracks = i->allTracks().size();
	  nAssociatedTracks->Fill(allTracks);
	  float selectedTracks = i->selectedTracks().size();
	  nSelectedTracks->Fill(selectedTracks);
	}	      
    }
  
    //Added By Konstantinos A. Petridis
    if(i->discriminator(0.1,0.07,0.4,6.,1.,0,0.2))
      {
	
	TLorentzVector recoTauJet(i->jet()->px(),i->jet()->py(),i->jet()->pz(),i->jet()->energy());
	bool trueTauJet=false;
	TLorentzVector* mcTauJetMatched=0;
	for(int m=0;m<TauJets->GetEntriesFast();m++)
	  {
	    TLorentzVector* trueTauJets=(TLorentzVector*)TauJets->At(m);
	    if(dR(trueTauJets,&recoTauJet)<0.2)
	      {
		trueTauJet=true;
		mcTauJetMatched=trueTauJets;
	      }
	  }
	
	if(trueTauJet)
	  {
	    double binSize=420.0/(1.0*nRuns);
	    double binSizeEta=2.2/(1.0*nRuns);
	    int EtBin=(mcTauJetMatched->Et())/binSize;
	    int EtaBin=(fabs(mcTauJetMatched->Eta()))/binSizeEta;
	    if(fabs(mcTauJetMatched->Eta())<2.2)
	      {
		ratio[EtBin]->Fill(recoTauJet.Et()/mcTauJetMatched->Et());
		if(mcTauJetMatched->Et()>30)ratioEta[EtaBin]->Fill(recoTauJet.Et()/mcTauJetMatched->Et());
	      }
	  
	    const TrackRef LeadingTrack3GeV=(i->leadingSignalTrack(0.1,3.0));
	    if(!LeadingTrack3GeV)cout<<"NO LEADING TRACK PASSES CRITERIA::"<<endl;
	    else
	      {
		math::XYZVector LeadingTrackMom=(*LeadingTrack3GeV).momentum();
		TVector3 LdgTrackMom3GeV(LeadingTrackMom.X(),LeadingTrackMom.Y(),LeadingTrackMom.Z());
		TVector3 RecoTauJetTight3GeV(i->jet()->px(),i->jet()->py(),i->jet()->pz());
		double LdgTrTauJetDR = Vec3dR(&LdgTrackMom3GeV,&RecoTauJetTight3GeV);
		if(mcTauJetMatched->Et()>40.0&&mcTauJetMatched->Et()<60)hDRRecLdgTrTauJet->Fill(LdgTrTauJetDR);
		if(mcTauJetMatched->Et()>200&&mcTauJetMatched->Et()<250)hDRRecLdgTrTauJet1->Fill(LdgTrTauJetDR);
	      }
	    
	  }
      }
    if(it1==0){
      TLorentzVector recTauJet(i->jet()->px(),i->jet()->py(),i->jet()->pz(),i->jet()->energy());
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
	  if(mcTauJetMtchd->Et()>130.0&&mcTauJetMtchd->Et()<150.0)
	    {
	      for(int ii=0;ii<20;ii++){
		nEventsUsed07[ii]++;
		double riso=ii*0.05;
		if(i->discriminator(0.1,0.07,riso,6.0,1.0,0,0.2)){
		  nEventsRiso07[ii]++;
		}
		if(i->discriminator(0.1,0.04,riso,6.0,1.0,0,0.2)){
		  nEventsRiso04[ii]++;
		}
	      }
	    }
	  if(mcTauJetMtchd->Et()>80.0&&mcTauJetMtchd->Et()<110.0)
	    {
	      for(int ii=0;ii<20;ii++){
		nEventsUsed107[ii]++;
		double riso=ii*0.05;
		if(i->discriminator(0.1,0.07,riso,6.0,1.0,0,0.2)){
		  nEventsRiso107[ii]++;
		}
		if(i->discriminator(0.1,0.04,riso,6.0,1.0,0,0.2)){
		  nEventsRiso104[ii]++;
		}
	      }
	    }
	  if(mcTauJetMtchd->Et()>50.0&&mcTauJetMtchd->Et()<70.0)
	    {
	      for(int ii=0;ii<20;ii++){
		nEventsUsed207[ii]++;
		double riso=ii*0.05;
		if(i->discriminator(0.1,0.07,riso,6.0,1.0,0,0.2)){
		  nEventsRiso207[ii]++;
		}
		if(i->discriminator(0.1,0.04,riso,6.0,1.0,0,0.2)){
		  nEventsRiso204[ii]++;
		}
	      }
	    }
	  if(mcTauJetMtchd->Et()>30.0&&mcTauJetMtchd->Et()<50.0)
	    {
	      for(int ii=0;ii<20;ii++){
		nEventsUsed307[ii]++;
		double riso=ii*0.05;
		if(i->discriminator(0.1,0.07,riso,6.0,1.0,0,0.2)){
		  nEventsRiso307[ii]++;
		}
		if(i->discriminator(0.1,0.04,riso,6.0,1.0,0,0.2)){
		  nEventsRiso304[ii]++;
		}
	      }
	    }
	}
    }
    it1++;
  }
  delete myGenEvent;
  delete TauJets;
}
  
void TauTagVal::endJob(){
  int ibin;
  for(int ii=0; ii<6; ii++){
    if(nEventsUsed[ii] > 0.)
      {
	ibin= ii+1;
	float eff= nEventsRiso[ii]/nEventsUsed[ii];
	effVsRiso->setBinContent(ibin,eff);
	float nEvents = 1.*nEventsRiso[ii];
	float nEventsTot = 1.*nEventsUsed[ii];
	EventseffVsRiso->setBinContent(ibin, nEvents);
	EventsToteffVsRiso->setBinContent(ibin, nEventsTot);
      }
    if(nEventsEnergyUsed[ii] > 0.)
      {
	ibin= ii+1;
	float eff= nEventsEnergy[ii]/nEventsEnergyUsed[ii];
	effVsEt->setBinContent(ibin,eff);
	float nEvents = 1.*nEventsEnergy[ii];
	float nEventsTot = 1.*nEventsEnergyUsed[ii];
	EventseffVsEt->setBinContent(ibin,nEvents);
	EventsToteffVsEt->setBinContent(ibin,nEventsTot);
      }
  }


  //Added By Konstantinos A. Petridis
  
  int ibin07=0,ibin107=0,ibin207=0,ibin307=0;
  int ibin04=0,ibin104=0,ibin204=0,ibin304=0;
  
  for(int ii=0; ii<20; ii++){
    
    if(nEventsUsed07[ii] > 0){ibin07= ii+1;ibin04= ii+1;}
    if(nEventsUsed107[ii] > 0){ibin107= ii+1;ibin104= ii+1;}
    if(nEventsUsed207[ii] > 0){ibin207= ii+1;ibin204= ii+1;}
    if(nEventsUsed307[ii] > 0){ibin307= ii+1;ibin304= ii+1;}
  
    if(nEventsUsed07[ii])
      {
	float effRiso= nEventsRiso07[ii]/nEventsUsed07[ii];
	float effRisoErr=sqrt(effRiso*(1-effRiso)/nEventsUsed07[ii]);
	effVsRiso07->setBinContent(ibin07,effRiso);
	effVsRiso07->setBinError(ibin07,effRisoErr); 
	float effRiso04= nEventsRiso04[ii]/nEventsUsed07[ii];
	float effRisoErr04=sqrt(effRiso04*(1-effRiso04)/nEventsUsed07[ii]);
	effVsRiso04->setBinContent(ibin04,effRiso04);
	effVsRiso04->setBinError(ibin04,effRisoErr04);
      }

    if(nEventsUsed107[ii])
      {   
	float effRiso1= nEventsRiso107[ii]/nEventsUsed107[ii];
	float effRisoErr1=sqrt(effRiso1*(1-effRiso1)/nEventsUsed107[ii]);
	effVsRiso107->setBinContent(ibin107,effRiso1);
	effVsRiso107->setBinError(ibin107,effRisoErr1); 
	float effRiso104= nEventsRiso104[ii]/nEventsUsed107[ii];
	float effRisoErr104=sqrt(effRiso104*(1-effRiso104)/nEventsUsed107[ii]);
	effVsRiso104->setBinContent(ibin104,effRiso104);
	effVsRiso104->setBinError(ibin104,effRisoErr104);

      }

    if(nEventsUsed207[ii])
      {
	float effRiso2= nEventsRiso207[ii]/nEventsUsed207[ii];
	float effRisoErr2=sqrt(effRiso2*(1-effRiso2)/nEventsUsed207[ii]);
	effVsRiso207->setBinContent(ibin207,effRiso2);
	effVsRiso207->setBinError(ibin207,effRisoErr2); 
	float effRiso204= nEventsRiso204[ii]/nEventsUsed207[ii];
	float effRisoErr204=sqrt(effRiso204*(1-effRiso204)/nEventsUsed207[ii]);
	effVsRiso204->setBinContent(ibin204,effRiso204);
	effVsRiso204->setBinError(ibin204,effRisoErr204);
	
      }	

    if(nEventsUsed307[ii])
      {
	float effRiso3= nEventsRiso307[ii]/nEventsUsed307[ii];
	float effRisoErr3=sqrt(effRiso3*(1-effRiso3)/nEventsUsed307[ii]);
	effVsRiso307->setBinContent(ibin307,effRiso3);
	effVsRiso307->setBinError(ibin307,effRisoErr3); 
	float effRiso304= nEventsRiso304[ii]/nEventsUsed307[ii];
	float effRisoErr304=sqrt(effRiso304*(1-effRiso304)/nEventsUsed307[ii]);
	effVsRiso304->setBinContent(ibin304,effRiso304);
	effVsRiso304->setBinError(ibin304,effRisoErr304);
    
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
      hEtmean->setBinContent(i+1,piEtmean[i]);
      hEtmean->setBinError(i,piEtmeanEr[i]);
      hEtamean->setBinContent(i+1,piEtamean[i]);
      hEtamean->setBinError(i+1,piEtmeanEr[i]);
    }


  effVsRiso07->setAxisRange(0.4,1.0,2);
  effVsRiso107->setAxisRange(0.4,1.0,2);
  effVsRiso207->setAxisRange(0.4,1.0,2);
  effVsRiso307->setAxisRange(0.4,1.0,2);
  effVsRiso04->setAxisRange(0.4,1.0,2);
  effVsRiso104->setAxisRange(0.4,1.0,2);
  effVsRiso204->setAxisRange(0.4,1.0,2);
  effVsRiso304->setAxisRange(0.4,1.0,2);
  hEtmean->setAxisRange(0.7,1.2,2);
  hEtmean->setAxisRange(0.7,1.2,2);


/*  
  effVsRiso07->GetXaxis()->SetTitle( "Isolation Cone" );
  effVsRiso07->GetYaxis()->SetTitle( "Efficiency" );
  effVsRiso107->GetXaxis()->SetTitle( "Isolation Cone" );
  effVsRiso107->GetYaxis()->SetTitle( "Efficiency" );
  effVsRiso207->GetXaxis()->SetTitle( "Isolation Cone" );
  effVsRiso207->GetYaxis()->SetTitle( "Efficiency" );
  effVsRiso307->GetXaxis()->SetTitle( "Isolation Cone" );
  effVsRiso307->GetYaxis()->SetTitle( "Efficiency" );

  effVsRiso04->GetXaxis()->SetTitle( "Isolation Cone" );
  effVsRiso04->GetYaxis()->SetTitle( "Efficiency" );
  effVsRiso104->GetXaxis()->SetTitle( "Isolation Cone" );
  effVsRiso104->GetYaxis()->SetTitle( "Efficiency" );
  effVsRiso204->GetXaxis()->SetTitle( "Isolation Cone" );
  effVsRiso204->GetYaxis()->SetTitle( "Efficiency" );
  effVsRiso304->GetXaxis()->SetTitle( "Isolation Cone" );
  effVsRiso304->GetYaxis()->SetTitle( "Efficiency" );
  */

  if (!outPutFile.empty() && &*edm::Service<DaqMonitorBEInterface>()) edm::Service<DaqMonitorBEInterface>()->save (outPutFile);
  
}
