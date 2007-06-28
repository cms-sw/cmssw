#ifndef TauTagVal_EMIso_BKG_h
#define TauTagVal_EMIso_BKG_h

// -*- C++ -*-
//
// Package:    TauTagVal_EMIso_BKG
// Class:      TauTagVal_EMIso_BKG
// 
/**\class TauTagVal_EMIso_BKG TauTagVal_EMIso_BKG.cc 

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
#include "DataFormats/BTauReco/interface/EMIsolatedTauTagInfo.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "HepMC/GenParticle.h"
// Math
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"
#include "TLorentzVector.h"
#include "TH1D.h"
#include <vector>

class MonitorElement;

//
// class decleration
//

    
class TauTagVal_EMIso_BKG : public edm::EDAnalyzer {
public:



  explicit TauTagVal_EMIso_BKG(const edm::ParameterSet&);
  ~TauTagVal_EMIso_BKG() {}

  virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);
  virtual void beginJob();
  virtual void endJob();
  
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
  edm::InputTag genJetSrc;
  std::vector<float> nEventsRiso;
  std::vector<float> nEventsEnergyUsed;
  std::vector<float> nEventsEnergy;
  std::vector<float> energyBins;
  
  std::vector<double>nEventsUsed07;
  std::vector<double>nEventsRiso07;

  std::vector<double>nEventsUsed107;
  std::vector<double>nEventsRiso107;


  std::vector<double>nEventsUsed207;
  std::vector<double>nEventsRiso207;

  std::vector<double>nEventsUsed307;
  std::vector<double>nEventsRiso307;



  TH1D* hRatio;
  TH1D* hRatioEta;
  std::vector<TH1D*> ratio;
  std::vector<TH1D*> ratioEta;
  std::vector<double> etbin;
  
  std::string outPutFile;
  float rSig,rMatch,ptLeadTk, rIso;
  MonitorElement* effVsRiso07;
  MonitorElement* effVsRiso107;
  MonitorElement* effVsRiso207;
  MonitorElement* effVsRiso307;


   

};

#endif











