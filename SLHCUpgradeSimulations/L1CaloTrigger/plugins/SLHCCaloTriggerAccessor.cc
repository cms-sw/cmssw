#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/SLHCCaloTriggerAccessor.h"

#include <iostream>
#include <iomanip>

using namespace reco;
using namespace l1slhc;
using namespace std;

SLHCCaloTriggerAccessor::SLHCCaloTriggerAccessor(const edm::ParameterSet& iConfig):
  l1egamma_     (iConfig.getParameter<edm::InputTag>("L1EGamma")),
  l1isoegamma_  (iConfig.getParameter<edm::InputTag>("L1IsoEGamma")),
  l1tau_        (iConfig.getParameter<edm::InputTag>("L1Tau")),
  l1isotau_     (iConfig.getParameter<edm::InputTag>("L1IsoTau")),
  jets_         (iConfig.getParameter<edm::InputTag>("Jets")),
  filename_     (iConfig.getParameter<std::string>("OutputFileName"))
{

// initialization

  f=new TFile(filename_.c_str(),"recreate");
  f->cd();
  t = new TTree("t","Tree");

// Variables

// Jets
  const int MaxJets = 200;
  jet_et   = new float[MaxJets];
  jet_eta  = new float[MaxJets];
  jet_phi  = new float[MaxJets];

// L1 extra
  const int MAXL1=15;

  nL1EG=0;
  l1eg_et    = new float[MAXL1];
  l1eg_eta   = new float[MAXL1];
  l1eg_phi   = new float[MAXL1];

  nL1IsoEG=0;
  l1isoeg_et   = new float[MAXL1];
  l1isoeg_eta  = new float[MAXL1];
  l1isoeg_phi  = new float[MAXL1];

  nL1Tau=0;
  l1tau_et   = new float[MAXL1];
  l1tau_eta  = new float[MAXL1];
  l1tau_phi  = new float[MAXL1];

  nL1IsoTau=0;
  l1isotau_et   = new float[MAXL1];
  l1isotau_eta  = new float[MAXL1];
  l1isotau_phi  = new float[MAXL1];

  //Branches
  t->Branch("L1EG_N",&nL1EG,"L1EG_N/I");
  t->Branch("L1EG_Et",l1eg_et,"L1EG_Et[L1EG_N]/F");
  t->Branch("L1EG_Eta",l1eg_eta,"L1EG_Eta[L1EG_N]/F");
  t->Branch("L1EG_Phi",l1eg_phi,"L1EG_Phi[L1EG_N]/F");

  t->Branch("L1ISOEG_N",&nL1IsoEG,"L1ISOEG_N/I");
  t->Branch("L1ISOEG_Et",l1isoeg_et,"L1ISOEG_Et[L1ISOEG_N]/F");
  t->Branch("L1ISOEG_Eta",l1isoeg_eta,"L1ISOEG_Eta[L1ISOEG_N]/F");
  t->Branch("L1ISOEG_Phi",l1isoeg_phi,"L1ISOEG_Phi[L1ISOEG_N]/F");

  t->Branch("L1Tau_N",&nL1Tau,"L1Tau_N/I");
  t->Branch("L1Tau_Et",l1tau_et,"L1Tau_Et[L1Tau_N]/F");
  t->Branch("L1Tau_Eta",l1tau_eta,"L1Tau_Eta[L1Tau_N]/F");
  t->Branch("L1Tau_Phi",l1tau_phi,"L1Tau_Phi[L1Tau_N]/F");

  t->Branch("L1IsoTau_N",&nL1IsoTau,"L1IsoTau_N/I");
  t->Branch("L1IsoTau_Et",l1isotau_et,"L1IsoTau_Et[L1IsoTau_N]/F");
  t->Branch("L1IsoTau_Eta",l1isotau_eta,"L1IsoTau_Eta[L1IsoTau_N]/F");
  t->Branch("L1IsoTau_Phi",l1isotau_phi,"L1IsoTau_Phi[L1IsoTau_N]/F");


  t->Branch("jet_N",&nJets,"jet_N/I");
  t->Branch("jet_Et",jet_et,"jet_Et[jet_N]/F");
  t->Branch("jet_Eta",jet_eta,"jet_Eta[jet_N]/F");
  t->Branch("jet_Phi",jet_phi,"jet_Phi[jet_N]/F");

}


SLHCCaloTriggerAccessor::~SLHCCaloTriggerAccessor()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
SLHCCaloTriggerAccessor::analyze(const edm::Event& iEvent,
                                 const edm::EventSetup& iSetup) {

  using namespace edm;
  using namespace reco;
  using namespace l1slhc;

// EGamma
  edm::Handle<l1extra::L1EmParticleCollection> l1egamma;
  if (iEvent.getByLabel(l1egamma_,l1egamma)) {
    int n=0;
    for (unsigned int i=0;i<l1egamma->size();i++) {
      l1eg_et[n]  = (*l1egamma)[i].et();
      l1eg_eta[n] = (*l1egamma)[i].eta();
      l1eg_phi[n] = (*l1egamma)[i].phi();
      n++;
    }
    nL1EG=n;
  }

// Iso EGamma
  edm::Handle<l1extra::L1EmParticleCollection> l1isoegamma;
  if (iEvent.getByLabel(l1isoegamma_,l1isoegamma)) {
    int n=0;
    for (unsigned int i=0;i<l1isoegamma->size();i++) {
      l1isoeg_et[n]  = (*l1isoegamma)[i].et();
      l1isoeg_eta[n] = (*l1isoegamma)[i].eta();
      l1isoeg_phi[n] = (*l1isoegamma)[i].phi();
      n++;
    }
    nL1IsoEG=n;
  }

// Tau
   edm::Handle<l1extra::L1JetParticleCollection> l1tau;
   if (iEvent.getByLabel(l1tau_,l1tau)) {
      int n=0;
      for (unsigned int i=0;i<l1tau->size();i++) {
        l1tau_et[n]  = (*l1tau)[i].et();
        l1tau_eta[n] = (*l1tau)[i].eta();
        l1tau_phi[n] = (*l1tau)[i].phi();
        n++;
      }
      nL1Tau=n;
    }

// ISDOTau
   edm::Handle<l1extra::L1JetParticleCollection> l1isotau;
   if (iEvent.getByLabel(l1isotau_,l1isotau)) {
      int n=0;
      for (unsigned int i=0;i<l1isotau->size();i++) {
	printf("Looping on Iso Tau\n");
        l1isotau_et[n]  = (*l1isotau)[i].et();
        l1isotau_eta[n] = (*l1isotau)[i].eta();
        l1isotau_phi[n] = (*l1isotau)[i].phi();
        n++;
      }
      nL1IsoTau=n;
    }
   else
     {
       printf("Iso Tau not found \n");
     }
// Jets
  edm::Handle<l1extra::L1JetParticleCollection> jets;
  int njets = 0;
  if (iEvent.getByLabel(jets_,jets)) {
    for (l1extra::L1JetParticleCollection::const_iterator i = jets->begin();i!=jets->end();++i) {
      jet_et[njets]  = i->et();
      jet_eta[njets] = i->eta();
      jet_phi[njets] = i->phi();
      njets++;
    }
    nJets = njets;
  }

  t->Fill();

}


// ------------ method called once each job just before starting event loop  ------------
void
SLHCCaloTriggerAccessor::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
SLHCCaloTriggerAccessor::endJob() {

  f->Write();
  f->Close();

}


