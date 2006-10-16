#ifndef TauTagVal_h
#define TauTagVal_h

// -*- C++ -*-
//
// Package:    TauTagVal
// Class:      TauTagVal
// 
/**\class TauTagVal TauTagVal.cc 

 Description: EDAnalyzer to validate the Collections from the ConeIsolation Producer
 It is supposed to be used for Offline Tau Reconstrction, so PrimaryVertex should be used.
 Implementation:
  
*/
//
// Original Author:  Simone Gennai
//
//



// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BTauReco/interface/JetTracksAssociation.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Math/interface/Vector3D.h"

// Math
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"

class MonitorElement;

//
// class decleration
//

class TauTagVal : public edm::EDAnalyzer {
public:
  explicit TauTagVal(const edm::ParameterSet&);
  ~TauTagVal() {}

  virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);
  virtual void beginJob();
  virtual void endJob();
private:
  int nEvent;
  vector<float> nEventsUsed;
  edm::InputTag jetTagSrc;
  vector<float> nEventsRiso;
  vector<float> nEventsEnergyUsed;
  vector<float> nEventsEnergy;
  vector<float> energyBins;
  std::string outPutFile;
  float rSig,rMatch,ptLeadTk, rIso;
  //AGGIUNGERE MC INFO???
  MonitorElement* effFindLeadTk;  
  MonitorElement* effVsRiso;
  MonitorElement* EventseffVsRiso;
  MonitorElement* EventsToteffVsRiso;
  MonitorElement* effVsEt;
  MonitorElement* EventseffVsEt;
  MonitorElement* EventsToteffVsEt;
  MonitorElement* nSignalTracks;
  MonitorElement* nSignalTracksAfterIsolation;
  MonitorElement* nAssociatedTracks;
  MonitorElement* nSelectedTracks;
  MonitorElement* ptLeadingTrack;
  MonitorElement* ptJet;
  MonitorElement* deltaRLeadTk_Jet;


};

#endif
