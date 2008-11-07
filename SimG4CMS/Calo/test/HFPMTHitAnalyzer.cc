// -*- C++ -*-
//
// Package:    HFPMTHitAnalyzer
// Class:      HFPMTHitAnalyzer
// 
/**\class HFPMTHitAnalyzer HFPMTHitAnalyzer.cc Analyzers/HFPMTHitAnalyzer/src/HFPMTHitAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Anthony Moeller
//         Created:  Wed Aug 27 10:18:19 CDT 2008
// $Id: HFPMTHitAnalyzer.cc,v 1.1 2008/10/29 18:58:10 sunanda Exp $
//
//


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//SimHits
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

//propagation
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

//other stuff
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h" 
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <math.h>
#include "TH1F.h"

#include "HFPMTHitAnalyzer.h"

HFPMTHitAnalyzer::HFPMTHitAnalyzer(const edm::ParameterSet& iConfig) {

  g4Label  = iConfig.getUntrackedParameter<std::string>("moduleLabel","g4SimHits");
  hcalHits = iConfig.getUntrackedParameter<std::string>("HitCollection","HcalHits");
}


HFPMTHitAnalyzer::~HFPMTHitAnalyzer() {}

void HFPMTHitAnalyzer::beginJob(const edm::EventSetup&) {

  event_no = 0;
  char name[20], title[120], sub[10];

  edm::Service<TFileService> fs;
  if ( !fs.isAvailable() )
    throw cms::Exception("BadConfig") << "TFileService unavailable: "
                                      << "please add it to config file";

  TFileDirectory HFHitsDir = fs->mkdir("HFPMTHits");
  h_HFDepHit = HFHitsDir.make<TH1F>("Hit20", "Depths in HF", 20, 0., 20.);
  h_HFDepHit->GetXaxis()->SetTitle("Depths in HF");
  TFileDirectory HFSourcePart = fs->mkdir("HFMCinfo");
  hHF_MC_e = HFSourcePart.make<TH1F>("MCEnergy","Energy of Generated Particle",1000,0.,500.);
  hHF_MC_e->GetXaxis()->SetTitle("Energy of Generated Particle");

  //energy Histograms
  TFileDirectory HFPCaloHitEnergyDir = fs->mkdir("HFPCaloHitEnergy"); 
  for (int i=0; i<3; i++) {
    if      (i == 0) sprintf (sub, "(Absorber)");
    else if (i == 1) sprintf (sub, "(PMT)");
    else             sprintf (sub, "(All)");
    sprintf (name,"Energy1%d",i); sprintf (title,"Energy in depth 1 %s",sub);
    hHF_e_1[i] = HFPCaloHitEnergyDir.make<TH1F>(name,title,1000,0.,500.);
    hHF_e_1[i]->GetXaxis()->SetTitle(title);
    sprintf (name,"Energy2%d",i); sprintf (title,"Energy in depth 2 %s",sub);
    hHF_e_2[i] = HFPCaloHitEnergyDir.make<TH1F>(name,title,1000,0.,500.);
    hHF_e_2[i]->GetXaxis()->SetTitle(title);
    sprintf (name,"Energy12%d",i); sprintf (title,"Energy in depths 1,2 %s",sub);
    hHF_e_12[i] = HFPCaloHitEnergyDir.make<TH1F>(name,title,1000,0.,500.);
    hHF_e_12[i]->GetXaxis()->SetTitle(title);
    sprintf (name,"Em1%d",i); sprintf (title,"EM energy in depth 1 %s",sub);
    hHF_em_1[i] = HFPCaloHitEnergyDir.make<TH1F>(name,title,1000,0.,500.);
    hHF_em_1[i]->GetXaxis()->SetTitle(title);
    sprintf (name,"Em2%d",i); sprintf (title,"EM energy in depth 2 %s",sub);
    hHF_em_2[i] = HFPCaloHitEnergyDir.make<TH1F>(name,title,1000,0.,500.);
    hHF_em_2[i]->GetXaxis()->SetTitle(title);
    sprintf (name,"Em12%d",i); sprintf (title,"EM energy in depths 1,2 %s",sub);
    hHF_em_12[i] = HFPCaloHitEnergyDir.make<TH1F>(name,title,1000,0.,500.);
    hHF_em_12[i]->GetXaxis()->SetTitle(title);
    sprintf (name,"Had1%d",i); sprintf (title,"Had energy in depth 1 %s",sub);
    hHF_had_1[i] = HFPCaloHitEnergyDir.make<TH1F>(name,title,1000,0.,0.1);
    hHF_had_1[i]->GetXaxis()->SetTitle(title);
    sprintf (name,"Had2%d",i); sprintf (title,"Had energy in depth 2 %s",sub);
    hHF_had_2[i] = HFPCaloHitEnergyDir.make<TH1F>(name,title,1000,0.,0.1);
    hHF_had_2[i]->GetXaxis()->SetTitle(title);
    sprintf (name,"Had12%d",i); sprintf (title,"Had energy in depths 1,2 %s",sub);
    hHF_had_12[i] = HFPCaloHitEnergyDir.make<TH1F>(name,title,1000,0.,0.1);
    hHF_had_12[i]->GetXaxis()->SetTitle(title);
  }

  //Timing Histograms
  TFileDirectory HFPCaloHitTimeDir = fs->mkdir("HFPCaloHitTime");
  for (int i=0; i<3; i++) {
    if      (i == 0) sprintf (sub, "(Absorber)");
    else if (i == 1) sprintf (sub, "(PMT)");
    else             sprintf (sub, "(All)");
    sprintf (name,"Time1Ewt%d",i); sprintf (title,"Time (energy weighted) in depth 1 %s",sub);
    hHF1_time_Ewt[i] = HFPCaloHitTimeDir.make<TH1F>(name,title,400,0.,400.);
    hHF1_time_Ewt[i]->GetXaxis()->SetTitle(title);
    sprintf (name,"Time2Ewt%d",i); sprintf (title,"Time (energy weighted) in depth 2 %s",sub);
    hHF2_time_Ewt[i] = HFPCaloHitTimeDir.make<TH1F>(name,title,400,0.,400.);
    hHF2_time_Ewt[i]->GetXaxis()->SetTitle(title);
    sprintf (name,"Time12Ewt%d",i); sprintf (title,"Time (energy weighted) in depths 1,2 %s",sub);
    hHF12_time_Ewt[i] = HFPCaloHitTimeDir.make<TH1F>(name,title,400,0.,400.);
    hHF12_time_Ewt[i]->GetXaxis()->SetTitle(title);
    sprintf (name,"Time1%d",i); sprintf (title,"Time in depth 1 %s",sub);
    hHF1_time[i] = HFPCaloHitTimeDir.make<TH1F>(name,title,400,0.,400.);
    hHF1_time[i]->GetXaxis()->SetTitle(title);
    sprintf (name,"Time2%d",i); sprintf (title,"Time in depth 2 %s",sub);
    hHF2_time[i] = HFPCaloHitTimeDir.make<TH1F>(name,title,400,0.,400.);
    hHF2_time[i]->GetXaxis()->SetTitle(title);
    sprintf (name,"Time12%d",i); sprintf (title,"Time in depths 1,2 %s",sub);
    hHF12_time[i] = HFPCaloHitTimeDir.make<TH1F>(name,title,400,0.,400.);
    hHF12_time[i]->GetXaxis()->SetTitle(title);
  }
}

void HFPMTHitAnalyzer::analyze(const edm::Event& iEvent, 
			       const edm::EventSetup& iSetup) {
  ++event_no; 
  if (event_no%500==0) std::cout<<"Event # "<<event_no<<" processed.\n";
  
 
  std::vector<PCaloHit> caloHits;
  edm::Handle<edm::PCaloHitContainer> hitsHcal;
  iEvent.getByLabel(g4Label,hcalHits,hitsHcal);
  std::vector<SimTrack> simTracks;
  edm::Handle<edm::SimTrackContainer> Tracks;
  iEvent.getByLabel (g4Label, Tracks);

  edm::Handle<edm::HepMCProduct > EvtHandle;
  iEvent.getByLabel("source", EvtHandle);
  const  HepMC::GenEvent* myGenEvent = EvtHandle->GetEvent();

  float orig_energy=0;
  for (HepMC::GenEvent::particle_const_iterator p=myGenEvent->particles_begin();
       p != myGenEvent->particles_end(); ++p ) {
 
    orig_energy=(*p)->momentum().e();
    hHF_MC_e->Fill(orig_energy);
  }

  caloHits.insert(caloHits.end(),hitsHcal->begin(),hitsHcal->end());
  analyzeHits (caloHits,*Tracks);

}

void HFPMTHitAnalyzer::analyzeHits (std::vector<PCaloHit>& hits,
				    const std::vector<SimTrack>& tracks1) {

  int nHit = hits.size();
  float energy1[3], energy2[3], energy12[3];
  float em1[3], had1[3], em2[3], had2[3], em12[3], had12[3];
  for (int i=0; i<3; i++) {
    energy1[i] = energy2[i] = energy12[i] = 0;
    em1[i]     = em2[i]     = em12[i]     = 0;
    had1[i]    = had2[i]    = had12[i]    = 0;
  }
  LogDebug("HFShower") << "HFPMTHitAnalyser::Entry " << event_no << " with "
		       << nHit << " hits";
  for (int i=0; i<nHit; i++)   {
    double energy    = hits[i].energy();
    double em        = hits[i].energyEM();
    double had       = hits[i].energyHad();
    double time      = hits[i].time();
    uint32_t id_     = hits[i].id();
    uint16_t  pmtHit = hits[i].depth();
    if (pmtHit != 0) pmtHit = 1;
    LogDebug("HFShower") << "HFPMTHitAnalyser::Hit " << i << " ID " << std::hex
			 << id_ << std::dec <<" PMT " << pmtHit <<" E (e|h|t) "
			 << em << " " << had << " " << energy <<  " Time " 
			 << time;

    HcalDetId id   = HcalDetId(id_);
    int det        = id.det();
    int subdet     = id.subdet();
    int depth      = id.depth();

    if (det ==  4) {
      
      if (subdet == static_cast<int>(HcalForward)) {
	h_HFDepHit->Fill(double(depth+10*pmtHit));
	if (depth==1)  {
	  energy1[pmtHit]  += energy;
	  energy12[pmtHit] += energy;
	  em1[pmtHit]      += em;
	  em12[pmtHit]     += em;
	  had1[pmtHit]     += had;
	  had12[pmtHit]    += had;
	  energy1[2]       += energy;
	  energy12[2]      += energy;
	  em1[2]           += em;
	  em12[2]          += em;
	  had1[2]          += had;
	  had12[2]         += had;
	  hHF1_time[pmtHit]->Fill(time);
	  hHF1_time[2]->Fill(time);
	  hHF1_time_Ewt[pmtHit]->Fill(time,energy);
	  hHF1_time_Ewt[2]->Fill(time,energy);
	}
	if (depth==2) {
	  energy2[pmtHit]  += energy;
	  energy12[pmtHit] += energy;
	  em2[pmtHit]      += em;
	  em12[pmtHit]     += em;
	  had2[pmtHit]     += had;
	  had12[pmtHit]    += had;
	  energy2[2]       += energy;
	  energy12[2]      += energy;
	  em2[2]           += em;
	  em12[2]          += em;
	  had2[2]          += had;
	  had12[2]         += had;
	  hHF2_time[pmtHit]->Fill(time);
	  hHF2_time[2]->Fill(time);
	  hHF2_time_Ewt[pmtHit]->Fill(time,energy);
	  hHF2_time_Ewt[2]->Fill(time,energy);
	}
      }
    }
  }
  for (int i=0; i<3; i++) {
    hHF_e_1[i]->Fill(energy1[i]);
    hHF_e_2[i]->Fill(energy2[i]);
    hHF_e_12[i]->Fill(energy12[i]);
    hHF_em_1[i]->Fill(em1[i]);
    hHF_em_2[i]->Fill(em2[i]);
    hHF_em_12[i]->Fill(em12[i]);
    hHF_had_1[i]->Fill(had1[i]);
    hHF_had_2[i]->Fill(had2[i]);
    hHF_had_12[i]->Fill(had12[i]);
    LogDebug("HFShower") << "HFPMTHitAnalyser:: Type " << i << " Energy 1|2| "
			 << energy1[i] << " " << energy2[i] << " "
			 << energy12[i] << " EM Energy 1|2| " << em1[i] << " "
			 << em2[i] << " " << em12[i] << " Had Energy 1|2| "
			 << had1[i] << " " << had2[i] << " " << had12[i];
  }
}

void HFPMTHitAnalyzer::endJob() {}

DEFINE_FWK_MODULE(HFPMTHitAnalyzer);
