#include "PhysicsTools/JetMCUtils/interface/combination.h"

#include "TopQuarkAnalysis/TopEventSelection/plugins/TtSemiLepSignalSelMVAComputer.h"
#include "TopQuarkAnalysis/TopEventSelection/interface/TtSemiLepSignalSelEval.h"

#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/PatCandidates/interface/Flags.h"


TtSemiLepSignalSelMVAComputer::TtSemiLepSignalSelMVAComputer(const edm::ParameterSet& cfg):
  muons_ (cfg.getParameter<edm::InputTag>("muons")),
  jets_    (cfg.getParameter<edm::InputTag>("jets")),
  METs_    (cfg.getParameter<edm::InputTag>("mets")),
  electrons_ (cfg.getParameter<edm::InputTag>("elecs"))
{
  produces< double >("DiscSel");
}

  

TtSemiLepSignalSelMVAComputer::~TtSemiLepSignalSelMVAComputer()
{
}

void
TtSemiLepSignalSelMVAComputer::produce(edm::Event& evt, const edm::EventSetup& setup)
{
  std::auto_ptr< double > pOutDisc (new double);
 
  mvaComputer.update<TtSemiLepSignalSelMVARcd>(setup, "ttSemiLepSignalSelMVA");

  // read name of the last processor in the MVA calibration
  // (to be used as meta information)
  edm::ESHandle<PhysicsTools::Calibration::MVAComputerContainer> calibContainer;
  setup.get<TtSemiLepSignalSelMVARcd>().get( calibContainer );
  std::vector<PhysicsTools::Calibration::VarProcessor*> processors
    = (calibContainer->find("ttSemiLepSignalSelMVA")).getProcessors();

  //make your preselection! This must!! be the same one as in TraintreeSaver.cc  
  edm::Handle<edm::View<pat::MET> > MET_handle;
  evt.getByLabel(METs_,MET_handle);
  if(!MET_handle.isValid()) return;
  const edm::View<pat::MET> MET = *MET_handle;

  edm::Handle< std::vector<pat::Jet> > jet_handle;
  evt.getByLabel(jets_, jet_handle);
  if(!jet_handle.isValid()) return;
  const std::vector<pat::Jet> jets = *jet_handle;
  unsigned int nJets = 0;
  std::vector<pat::Jet> seljets;
  for(std::vector<pat::Jet>::const_iterator it = jets.begin(); it != jets.end(); it++) {
    if(!(pat::Flags::test(*it, pat::Flags::Overlap::Electrons))) continue;
    if(it->pt()>30. && fabs(it->eta())<2.4) {
      seljets.push_back(*it);
      nJets++;
    }
  }
   
  edm::Handle< edm::View<pat::Muon> > muon_handle; 
  evt.getByLabel(muons_, muon_handle);
  if(!muon_handle.isValid()) return;
  const edm::View<pat::Muon> muons = *muon_handle;
  int nmuons = 0;
  std::vector<pat::Muon> selMuons;
  for(edm::View<pat::Muon>::const_iterator it = muons.begin(); it!=muons.end(); it++) {
    reco::TrackRef gltr = it->track(); // global track
    reco::TrackRef trtr = it->innerTrack(); // tracker track
    if(it->pt()>30 && fabs(it->eta())<2.1 && (it->pt()/(it->pt()+it->trackIso()+it->caloIso()))>0.95 && it->isGlobalMuon()){
      if(gltr.isNull()) continue;  //temporary problems with dead trackrefs
      if((gltr->chi2()/gltr->ndof())<10 && trtr->numberOfValidHits()>11) {
        double dRmin = 9999.;
        for(std::vector<pat::Jet>::const_iterator ajet = seljets.begin(); ajet != seljets.end(); ajet++) {
          math::XYZTLorentzVector jet = ajet->p4();
          math::XYZTLorentzVector muon = it->p4();
          double tmpdR = DeltaR(muon,jet);
          if(tmpdR<dRmin) dRmin = tmpdR;
        }
        if(dRmin>0.3) {   //temporary problems with muon isolation
          nmuons++;
          selMuons.push_back(*it);
        }
      }
    }
  }
  
  edm::Handle< edm::View<pat::Electron> > electron_handle; 
  evt.getByLabel(electrons_, electron_handle);
  if(!electron_handle.isValid()) return;
  const edm::View<pat::Electron> electrons = *electron_handle;
  int nelectrons = 0;
  for(edm::View<pat::Electron>::const_iterator it = electrons.begin(); it!=electrons.end(); it++) {
    if(it->pt()>30 && fabs(it->eta())<2.4 && (it->pt()/(it->pt()+it->trackIso()+it->caloIso()))>0.95 && it->isElectronIDAvailable("eidTight"))
    { 
      if(it->electronID("eidTight")==1) nelectrons++;
    }
  }
 
  
  double discrim;
  // discriminator output for events which do not pass the preselection is set to -1
  if( nmuons!=1                    ||
      nJets < 4                    ||
      nelectrons > 0 ) discrim = -1.; //std::cout<<"nJets: "<<seljets.size()<<" numLeptons: "<<nleptons<<std::endl;}
  else {
    //check wheter a event was already selected (problem with duplicated events)
    math::XYZTLorentzVector muon = selMuons.begin()->p4();

    TtSemiLepSignalSel selection(seljets,muon,MET);

    discrim = evaluateTtSemiLepSignalSel(mvaComputer, selection);
  }

  *pOutDisc = discrim;
  
  evt.put(pOutDisc, "DiscSel");
  
  DiscSel = discrim;

}

void 
TtSemiLepSignalSelMVAComputer::beginJob()
{
}

void 
TtSemiLepSignalSelMVAComputer::endJob()
{
}

double TtSemiLepSignalSelMVAComputer::DeltaPhi(const math::XYZTLorentzVector& v1, const math::XYZTLorentzVector& v2)
{
  double dPhi = fabs(v1.Phi() - v2.Phi());
  if (dPhi > TMath::Pi()) dPhi =  2*TMath::Pi() - dPhi;
  return dPhi;
}

double TtSemiLepSignalSelMVAComputer::DeltaR(const math::XYZTLorentzVector& v1, const math::XYZTLorentzVector& v2)
{
  double dPhi = DeltaPhi(v1,v2);
  double dR = TMath::Sqrt((v1.Eta()-v2.Eta())*(v1.Eta()-v2.Eta())+dPhi*dPhi);
  return dR;
}

// implement the plugins for the computer container
// -> register TtSemiLepSignalSelMVARcd
// -> define TtSemiLepSignalSelMVAFileSource
MVA_COMPUTER_CONTAINER_IMPLEMENT(TtSemiLepSignalSelMVA);
