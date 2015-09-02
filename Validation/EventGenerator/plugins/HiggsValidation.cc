/*class HiggsValidation
 *  
 *  Class to fill dqm monitor elements from existing EDM file
 *
 */
 
#include "Validation/EventGenerator/interface/HiggsValidation.h"

#include "FWCore/Framework/interface/MakerMacros.h"  

#include "CLHEP/Units/defs.h"
#include "CLHEP/Units/PhysicalConstants.h"

#include "DataFormats/Math/interface/LorentzVector.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "Validation/EventGenerator/interface/PdtPdgMini.h"
#include "Validation/EventGenerator/interface/DQMHelper.h"
using namespace edm;

HiggsValidation::HiggsValidation(const edm::ParameterSet& iPSet): 
  wmanager_(iPSet,consumesCollector()),
  hepmcCollection_(iPSet.getParameter<edm::InputTag>("hepmcCollection")),
  particle_id(iPSet.getParameter<int>("pdg_id")),
  particle_name(iPSet.getParameter<std::string>("particleName"))
{    
  monitoredDecays = new MonitoredDecays(iPSet);
  hepmcCollectionToken_=consumes<HepMCProduct>(hepmcCollection_);
}

HiggsValidation::~HiggsValidation() {}

void HiggsValidation::dqmBeginRun(const edm::Run& r, const edm::EventSetup& c) {
  c.getData( fPDGTable );
}

void HiggsValidation::bookHistograms(DQMStore::IBooker &i, edm::Run const &, edm::EventSetup const &){

    ///Setting the DQM top directories
    TString dir="Generator/";
    dir+=particle_name;
    DQMHelper dqm(&i); i.setCurrentFolder(dir.Data());
    
    // Number of analyzed events
    nEvt = dqm.book1dHisto("nEvt", "n analyzed Events", 1, 0., 1.);
    
    //decay type
    
    std::string channel = particle_name+"_DecayChannels";
    HiggsDecayChannels = dqm.book1dHisto(channel.c_str(),(particle_name+" decay channels").c_str(),monitoredDecays->size(),0,monitoredDecays->size());
    
    for(size_t j = 0; j < monitoredDecays->size(); ++j){
      HiggsDecayChannels->setBinLabel(1+j,monitoredDecays->channel(j));
    }
  

  //Kinematics 
  Higgs_pt = dqm.book1dHisto((particle_name+"_pt"),(particle_name+" p_{t}"),50,0,250);
  Higgs_eta  = dqm.book1dHisto((particle_name+"_eta"),(particle_name+" #eta"),50,-5,5); 
  Higgs_mass = dqm.book1dHisto((particle_name+"_m"),(particle_name+" M"),500,0,500);

  int idx=0;
  for(unsigned int j=0;j<monitoredDecays->NDecayParticles();j++){
    HiggsDecayProd_pt.push_back(dqm.book1dHisto((monitoredDecays->ConvertIndex(idx)+"_pt"),(monitoredDecays->ConvertIndex(idx)+" p_{t}"),50,0,250));
    HiggsDecayProd_eta.push_back(dqm.book1dHisto((monitoredDecays->ConvertIndex(idx)+"_eta"),(monitoredDecays->ConvertIndex(idx)+" #eta"),50,-5,5));
    idx++;
  }

  return;
}

void HiggsValidation::analyze(const edm::Event& iEvent,const edm::EventSetup& iSetup)
{ 
  double weight =   wmanager_.weight(iEvent);  
  nEvt->Fill(0.5,weight);
  
  //Gathering the HepMCProduct information
  edm::Handle<HepMCProduct> evt;
  iEvent.getByToken(hepmcCollectionToken_, evt);
  
  //Get EVENT
  HepMC::GenEvent *myGenEvent = new HepMC::GenEvent(*(evt->GetEvent()));

  // loop over all particles
  bool filled = false;
  for(HepMC::GenEvent::particle_const_iterator iter = myGenEvent->particles_begin(); 
      iter!= myGenEvent->particles_end() && !filled; ++iter) {
    if(particle_id == fabs((*iter)->pdg_id())){
      std::vector<HepMC::GenParticle*> decayprod;
      int channel = findHiggsDecayChannel(*iter,decayprod);
      HiggsDecayChannels->Fill(channel,weight);
      Higgs_pt->Fill((*iter)->momentum().perp(),weight); 
      Higgs_eta->Fill((*iter)->momentum().eta(),weight);
      Higgs_mass->Fill((*iter)->momentum().m(),weight);
      for(unsigned int i=0;i<decayprod.size();i++){
	int idx=monitoredDecays->isDecayParticle(decayprod.at(i)->pdg_id());
	if(0<=idx && idx<=(int)HiggsDecayProd_pt.size()){
	  HiggsDecayProd_pt.at(idx)->Fill(decayprod.at(i)->momentum().perp(),weight);
	  HiggsDecayProd_eta.at(idx)->Fill(decayprod.at(i)->momentum().eta(),weight);
	}
      }
      filled = true;
    }
  }
  
  delete myGenEvent;
  
}//analyze

int HiggsValidation::findHiggsDecayChannel(const HepMC::GenParticle* genParticle,std::vector<HepMC::GenParticle*> &decayprod){
  if(genParticle->status() == 1) return monitoredDecays->stable();
  std::vector<int> children;
  if ( genParticle->end_vertex() ) {
    HepMC::GenVertex::particle_iterator des;
    for(des = genParticle->end_vertex()->particles_begin(HepMC::descendants);
	des!= genParticle->end_vertex()->particles_end(HepMC::descendants);++des ) {
      
      if((*des)->pdg_id() == genParticle->pdg_id()) continue;
      
      HepMC::GenVertex::particle_iterator mother = (*des)->production_vertex()->particles_begin(HepMC::parents);
      if((*mother)->pdg_id() == genParticle->pdg_id()){
	children.push_back((*des)->pdg_id());
	decayprod.push_back((*des));
      }
    }
  }
  
  if(children.size() == 2 && children.at(0) != 0 && children.at(1) != 0){
    return monitoredDecays->position(children.at(0),children.at(1));
  }
  return monitoredDecays->undetermined();
}
