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
#include "HepMC/GenParticle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BTauReco/interface/JetTracksAssociation.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

// Math
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"
#include "TLorentzVector.h"
#include "TH1D.h"
#include <vector>
namespace HepMC {
  class GenParticle;
 }
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
  /*
  std::vector<HepMC::GenParticle*> DaughtersVec(std::vector<HepMC::GenParticle*> pvec)
  {
    std::vector<HepMC::GenParticle*>Daughters;
    HepMC::GenVertex* end_vertex;

    for(std::vector<HepMC::GenParticle*>::iterator pvecit=pvec.begin();pvecit!=pvec.end();pvecit++)
      {
	end_vertex = (*pvecit)->end_vertex();
	if(end_vertex){
	  HepMC::GenVertex::particles_out_const_iterator outp;
	  for(outp=end_vertex->particles_out_const_begin();outp!=end_vertex->particles_out_const_end();++outp)
	    {
	      Daughters.push_back((*outp));
	    }
	}
      }
    return Daughters;
  }
  */
  std::vector<HepMC::GenParticle*> Daughters(HepMC::GenParticle* p)
  {
    std::vector<HepMC::GenParticle*>Daughters;
    HepMC::GenVertex* end_vertex;
    end_vertex = p->end_vertex();
    if(end_vertex)
      {
	HepMC::GenVertex::particles_out_const_iterator outp;
	for(outp=end_vertex->particles_out_const_begin();outp!=end_vertex->particles_out_const_end();++outp)
	  {
	    Daughters.push_back((*outp));
	  }
      }
    return Daughters;
  }  

  double dR(TLorentzVector* v1,TLorentzVector* v2)
  {
    double DR;
    double dphi=fabs(v1->Phi()-v2->Phi());
    if(dphi>acos(-1.0))dphi=2*acos(-1.0)-dphi;
    double deta=fabs(v1->Eta()-v2->Eta());
    DR=sqrt(dphi*dphi+deta*deta);
    return DR;
  }

  double Vec3dR(TVector3* v1,TVector3* v2)
  {
    double DR;
    double dphi=fabs(v1->Phi()-v2->Phi());
    if(dphi>acos(-1.0))dphi=2*acos(-1.0)-dphi;
    double deta=fabs(v1->Eta()-v2->Eta());
    DR=sqrt(dphi*dphi+deta*deta);
    return DR;
  }

private:
  int nEvent;
  int nRuns;
  std::vector<float> nEventsUsed;
  edm::InputTag jetTagSrc;
  std::vector<float> nEventsRiso;
  std::vector<float> nEventsEnergyUsed;
  std::vector<float> nEventsEnergy;
  std::vector<float> energyBins;
  
  std::vector<double>nEventsUsed07;
  std::vector<double>nEventsRiso07;
  std::vector<double>nEventsUsed04;
  std::vector<double>nEventsRiso04;

  std::vector<double>nEventsUsed107;
  std::vector<double>nEventsRiso107;
  std::vector<double>nEventsUsed104;
  std::vector<double>nEventsRiso104;

  std::vector<double>nEventsUsed207;
  std::vector<double>nEventsRiso207;
  std::vector<double>nEventsUsed204;
  std::vector<double>nEventsRiso204;

  std::vector<double>nEventsUsed307;
  std::vector<double>nEventsRiso307;
  std::vector<double>nEventsUsed304;
  std::vector<double>nEventsRiso304;


  TH1D* hRatio;
  TH1D* hRatioEta;
  std::vector<TH1D*> ratio;
  std::vector<TH1D*> ratioEta;
  std::vector<double> etbin;
  
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
  MonitorElement* hEtmean;
  MonitorElement* hEtamean;
  MonitorElement *hDRRecLdgTrTauJet;
  MonitorElement *hDRRecLdgTrTauJet1;
  MonitorElement* effVsRiso07;
  MonitorElement* effVsRiso107;
  MonitorElement* effVsRiso207;
  MonitorElement* effVsRiso307;
  MonitorElement* effVsRiso04;
  MonitorElement* effVsRiso104;
  MonitorElement* effVsRiso204;
  MonitorElement* effVsRiso304;
  MonitorElement* hTauJets;
   

};

#endif











