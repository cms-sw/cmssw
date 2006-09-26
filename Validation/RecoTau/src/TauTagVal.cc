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
// $Id: TauTagVal.cc,v 1.12 2006/07/17 15:12:35 gennai Exp $
//
//


// user include files
#include "Validation/RecoTau/interface/TauTagVal.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/interface/MonitorElement.h"

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
  nEventsRiso.reserve(6);
  nEventsUsed.reserve(6);
  for(int i=0;i<6;i++){
    nEventsRiso[i]=0.;
    nEventsUsed[i]=0.;
  }
  
  DaqMonitorBEInterface* dbe = &*edm::Service<DaqMonitorBEInterface>();
  if(dbe) {
    dbe->setCurrentFolder("TauJetTask_" + jetTagSrc.label());    
    ptLeadingTrack = dbe->book1D("PtLeadTk", "Pt LeadTk", 100, 0., 300.);
    ptJet  = dbe->book1D("PtJet", "Pt Jet", 100, 0., 300.);
    nSignalTracks = dbe->book1D("NSigTks", "NSigTks", 10, 0., 10.);
    effVsRiso = dbe->book1D("Eff","Eff",6,0.2,0.5);
  }
  
  if (outPutFile.empty ()) {
    LogInfo("OutputInfo") << " TauJet histograms will NOT be saved";
  } 
  else {
    LogInfo("OutputInfo") << " TauJethistograms will be saved to file:" << outPutFile;
  }
  
}

void TauTagVal::beginJob(){ }


void TauTagVal::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace reco;
  
  Handle<IsolatedTauTagInfoCollection> tauTagInfoHandle;
  iEvent.getByLabel(jetTagSrc, tauTagInfoHandle);
  
  const IsolatedTauTagInfoCollection & tauTagInfo = *(tauTagInfoHandle.product());
  
  IsolatedTauTagInfoCollection::const_iterator i = tauTagInfo.begin();
  int it=0;
  for (; i != tauTagInfo.end(); ++i) {
    //To compute the efficiency as a function of the isolation cone (Riso)
    if(it == 0) {
      for(int ii=0;ii<6;ii++)
	{
	  nEventsUsed[ii]++;
	  float Riso = ii*0.05 + 0.2;
	  float Rmatch = 0.1;
	  float Rsig = 0.07;
	  float pT_LT = 6.;
	  float pT_min =1.;
	  if( i->discriminator(Rmatch,Rsig,Riso,pT_LT,pT_min) > 0) nEventsRiso[ii]++;
	}
      const TrackRef leadTk= (i->leadingSignalTrack(rMatch, 1.));
      if(!leadTk){
	LogInfo("LeadingTrack") << " No LeadingTrack";
      }else{
	ptLeadingTrack->Fill((*leadTk).pt());
	ptJet->Fill((i->jet()).pt());
	math::XYZVector momentum = (*leadTk).momentum();
	float nsigtks = (i->tracksInCone(momentum, rSig,  1.)).size();
	nSignalTracks->Fill(nsigtks);
      }
    }
  }
}

void TauTagVal::endJob(){
  int ibin;
  for(int ii=0; ii<6; ii++){
    if(nEventsUsed[ii] > 0)
      ibin= ii+1;
    float eff= nEventsRiso[ii]/nEventsUsed[ii];
    effVsRiso->setBinContent(ibin,eff);
  }
  
  if (!outPutFile.empty() && &*edm::Service<DaqMonitorBEInterface>()) edm::Service<DaqMonitorBEInterface>()->save (outPutFile);
  
}
