#include "Validation/EventGenerator/interface/TTbar_Kinematics.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "Validation/EventGenerator/interface/PdtPdgMini.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "Validation/EventGenerator/interface/DQMHelper.h"

using namespace edm;
TTbar_Kinematics::TTbar_Kinematics(const edm::ParameterSet& iConfig) :
  hepmcCollection_(iConfig.getParameter<edm::InputTag>("hepmcCollection"))
  ,genEventInfoProductTag_(iConfig.getParameter<edm::InputTag>("genEventInfoProductTag"))
{

  hepmcCollectionToken_=consumes<HepMCProduct>(hepmcCollection_);
  genEventInfoProductTagToken_=consumes<GenEventInfoProduct>(genEventInfoProductTag_);
}


TTbar_Kinematics::~TTbar_Kinematics(){}


//
// member functions
//

// ------------ method called for each event  ------------
void TTbar_Kinematics::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  // --- the MC weights ---
  Handle<GenEventInfoProduct> evt_info;
  iEvent.getByToken(genEventInfoProductTagToken_, evt_info);
  if(!evt_info.isValid()) return;
  weight = evt_info->weight() ;



  ///Gathering the HepMCProduct information
  edm::Handle<HepMCProduct> evt;
  iEvent.getByToken(hepmcCollectionToken_, evt);

  //Get EVENT
  const HepMC::GenEvent *myGenEvent = evt->GetEvent();

  TLorentzVector tlv_Top, tlv_TopBar, tlv_Bottom, tlv_BottomBar ,tlv_Wplus ,tlv_Wmin , tlv_TTbar;
  bool top(false), antitop(false), antibottom(false), bottom(false), Wplus(false), Wmin(false);
  for(HepMC::GenEvent::particle_const_iterator iter = myGenEvent->particles_begin(); iter != myGenEvent->particles_end(); iter++) {
    if((*iter)->pdg_id()==PdtPdgMini::t || (*iter)->pdg_id()==PdtPdgMini::anti_t){
      if( (*iter)->end_vertex()){
	HepMC::GenVertex::particle_iterator des;
	for(des = (*iter)->end_vertex()->particles_begin(HepMC::children);des!= (*iter)->end_vertex()->particles_end(HepMC::children);++des ){
	  if((*des)->pdg_id()==PdtPdgMini::b){
	    tlv_Bottom.SetPxPyPzE((*des)->momentum().px(),(*des)->momentum().py(),(*des)->momentum().pz(),(*des)->momentum().e());
	    bottom=true;
 	  }
          if((*des)->pdg_id()==PdtPdgMini::anti_b){
	    antibottom=true;
	    tlv_BottomBar.SetPxPyPzE((*des)->momentum().px(),(*des)->momentum().py(),(*des)->momentum().pz(),(*des)->momentum().e());
         }
	  if((*des)->pdg_id()==PdtPdgMini::W_plus){ 
	    tlv_TopBar.SetPxPyPzE((*iter)->momentum().px(),(*iter)->momentum().py(),(*iter)->momentum().pz(),(*iter)->momentum().e()); antitop=true;
	    tlv_Wplus.SetPxPyPzE((*des)->momentum().px(),(*des)->momentum().py(),(*des)->momentum().pz(),(*des)->momentum().e()); Wplus=true;
	  }
	  if((*des)->pdg_id()==PdtPdgMini::W_minus){ 
	    tlv_Top.SetPxPyPzE((*iter)->momentum().px(),(*iter)->momentum().py(),(*iter)->momentum().pz(),(*iter)->momentum().e()); top=true;
	    tlv_Wmin.SetPxPyPzE((*des)->momentum().px(),(*des)->momentum().py(),(*des)->momentum().pz(),(*des)->momentum().e()); Wmin=true;
	  }
	}
      }
    }
  }
	  
  tlv_TTbar = tlv_Top + tlv_TopBar ;

  //---topquarkquantities---
  nEvt->Fill(0.5,weight);
  if(top && antitop){
    hTopPt->Fill(tlv_Top.Pt(),weight);
    hTopPt->Fill(tlv_TopBar.Pt(),weight);
    
    hTopY->Fill(tlv_Top.Rapidity(),weight);
    hTopY->Fill(tlv_TopBar.Rapidity(),weight);
    
    hTopMass->Fill(tlv_Top.M(),weight);
    hTopMass->Fill(tlv_TopBar.M(),weight);
    
    //---ttbarpairquantities---
    hTTbarPt->Fill(tlv_TTbar.Pt(),weight);
    hTTbarY->Fill(tlv_TTbar.Rapidity(),weight);
    hTTbarMass->Fill(tlv_TTbar.M(),weight);
  }
  if(bottom && antibottom){
    hBottomPt->Fill(tlv_Bottom.Pt(),weight);
    hBottomPt->Fill(tlv_BottomBar.Pt(),weight);
    
    hBottomEta->Fill(tlv_Bottom.Eta(),weight);
    hBottomEta->Fill(tlv_BottomBar.Eta(),weight);
    
    //hBottomY->Fill(math::XYZTLorentzVector(bottom->momentum()).Rapidity(),weight);
    //hBottomY->Fill(math::XYZTLorentzVector(antibottom->momentum()).Rapidity(),weight);
    
    hBottomY->Fill(tlv_Bottom.Rapidity(),weight);
    hBottomY->Fill(tlv_BottomBar.Rapidity(),weight);
    
    hBottomPz->Fill(tlv_Bottom.Pz(),weight);
    hBottomPz->Fill(tlv_BottomBar.Pz(),weight);
    
    hBottomE->Fill(tlv_Bottom.E(),weight);
    hBottomE->Fill(tlv_BottomBar.E(),weight);
    
    hBottomMass->Fill(tlv_Bottom.M(),weight);
    hBottomMass->Fill(tlv_BottomBar.M(),weight);
    
    hBottomPtPz->Fill(tlv_Bottom.Pt(),tlv_Bottom.Pz(),weight);
    hBottomPtPz->Fill(tlv_BottomBar.Pt(),tlv_BottomBar.Pz(),weight);
    
    hBottomEtaPz->Fill(tlv_Bottom.Eta(),tlv_Bottom.Pz(),weight);
    hBottomEtaPz->Fill(tlv_BottomBar.Eta(),tlv_BottomBar.Pz(),weight);
    
    hBottomEtaPt->Fill(tlv_Bottom.Eta(),tlv_Bottom.Pt(),weight);
    hBottomEtaPt->Fill(tlv_BottomBar.Eta(),tlv_BottomBar.Pt(),weight);
    
    hBottomYPz->Fill(tlv_Bottom.Rapidity(),tlv_Bottom.Pz(),weight);
    hBottomYPz->Fill(tlv_BottomBar.Rapidity(),tlv_BottomBar.Pz(),weight);
    
    hBottomMassPz->Fill(tlv_Bottom.M(),tlv_Bottom.Pz(),weight);
    hBottomMassPz->Fill(tlv_BottomBar.M(),tlv_BottomBar.Pz(),weight);
    
    hBottomMassEta->Fill(tlv_Bottom.M(),tlv_Bottom.Eta(),weight);
    hBottomMassEta->Fill(tlv_BottomBar.M(),tlv_BottomBar.Eta(),weight);
    
    hBottomMassY->Fill(tlv_Bottom.M(),tlv_Bottom.Rapidity(),weight);
    hBottomMassY->Fill(tlv_BottomBar.M(),tlv_BottomBar.Rapidity(),weight);
    
    hBottomMassDeltaY->Fill(tlv_Bottom.M(),tlv_Bottom.Eta()-tlv_Bottom.Rapidity(),weight);
    hBottomMassDeltaY->Fill(tlv_BottomBar.M(),tlv_BottomBar.Eta()-tlv_BottomBar.Rapidity(),weight);
  }
  if(Wplus && Wmin){
    hWplusPz->Fill(tlv_Wplus.Pz(),weight);
    hWminPz->Fill(tlv_Wmin.Pz(),weight);
  }
}


// ------------ method called once each job just before starting event loop  ------------
void TTbar_Kinematics::bookHistograms(DQMStore::IBooker &i, edm::Run const &r, edm::EventSetup const &e){
  DQMHelper dqm(&i); i.setCurrentFolder("Generator/TTbar");

  nEvt = dqm.book1dHisto("nEvt", "n analyzed Events", 1, 0., 1.);

  hTopPt         = dqm.book1dHisto("TTbar_TopPt","t quark transverse momentum",1000,0.,1000.); hTopPt->setAxisTitle("t quark transverse momentum");
  hTopY          = dqm.book1dHisto("TTbar_TopY","t quark rapidity",200,-5.,5.);                hTopY->setAxisTitle("t quark rapidity");
  hTopMass       = dqm.book1dHisto("TTbar_TopMass","t quark mass",500,0.,500.);                hTopMass->setAxisTitle("t quark mass");
 
  hTTbarPt       = dqm.book1dHisto("TTbar_TTbarPt","tt pair transverse momentum",1000,0.,1000.); hTTbarPt->setAxisTitle("tt pair transverse momentum");
  hTTbarY        = dqm.book1dHisto("TTbar_TTbarY","tt pair rapidity",200,-5.,5.);                hTTbarY->setAxisTitle("tt pair rapidity");
  hTTbarMass     = dqm.book1dHisto("TTbar_TTbarMass","tt pair mass",1000,0.,1000.);              hTTbarMass->setAxisTitle("tt pair mass");

  hBottomPt      = dqm.book1dHisto("TTbar_BottomPt","b quark transverse momentum",1000,0.,1000.);     hBottomPt->setAxisTitle("b quark transverse momentum");
  hBottomEta     = dqm.book1dHisto("TTbar_BottomEta","b quark pseudo-rapidity",200,-5.,5.);           hBottomEta->setAxisTitle("b quark pseudo-rapidity");
  hBottomY       = dqm.book1dHisto("TTbar_BottomY","b quark rapidity",200,-5.,5.);                    hBottomY->setAxisTitle("b quark rapidity");
  hBottomPz      = dqm.book1dHisto("TTbar_BottomPz","b quark longitudinal momentum",200,-100.,100.);  hBottomPz->setAxisTitle("b quark longitudinal momentum");
  hBottomE       = dqm.book1dHisto("TTbar_BottomE","b quark energy",1000,0.,1000.);                   hBottomE->setAxisTitle("b quark energy");
  hBottomMass    = dqm.book1dHisto("TTbar_BottomMass","b quark mass",50,0.,5.);                       hBottomMass->setAxisTitle("b quark mass");
  
  hBottomPtPz    = dqm.book2dHisto("TTbar_BottomPtPz","b quark longitudinal vs transverse momentum",1000,0.,1000.,200,-100.,100.);     hBottomPtPz->setAxisTitle("P_{z} (GeV)",1);hBottomPtPz->setAxisTitle("P_{t} (GeV)",2);
  hBottomEtaPz   = dqm.book2dHisto("TTbar_BottomEtaPz","b quark longitudinal momentum vs pseudorapidity",200,-5.,5.,200,-100.,100.);   hBottomEtaPz->setAxisTitle("#eta",1);hBottomEtaPz->setAxisTitle("P_{z} (GeV)",1);
  hBottomEtaPt   = dqm.book2dHisto("TTbar_BottomEtaPt"," quark transveral   momentum vs pseudorapidity",200,-5.,5.,1000,0.,1000.);   hBottomEtaPt->setAxisTitle("#eta");hBottomEtaPt->setAxisTitle("P_{t} (GeV)");
  hBottomYPz     = dqm.book2dHisto("TTbar_BottomYPz","b quark longitudinal momentum vs rapidity",200,-5.,5.,200,-100.,100.);           hBottomYPz->setAxisTitle("Y");hBottomYPz->setAxisTitle("P_{z} (GeV)");
  hBottomMassPz  = dqm.book2dHisto("TTbar_BottomMassPz","b quark longitudinal momentum vs mass",50,0.,5.,200,-100.,100.);              hBottomMassPz->setAxisTitle("M (GeV)");hBottomMassPz->setAxisTitle("P_{z} (GeV)");
  hBottomMassEta = dqm.book2dHisto("TTbar_BottomMassEta","b quark pseudorapidity vs mass",50,0.,5.,200,-5.,5.);                        hBottomMassEta->setAxisTitle("M (GeV)");hBottomMassEta->setAxisTitle("#eta");
  hBottomMassY   = dqm.book2dHisto("TTbar_BottomMassY","b quark rapidity vs mass",50,0.,5.,200,-5.,5.);                                hBottomMassY->setAxisTitle("M (GeV)"); hBottomMassY->setAxisTitle("Y");
  hBottomMassDeltaY = dqm.book2dHisto("TTbar_BottomMassDeltaY","b quark pseudorapidity - rapidity vs mass",50,0.,50.,2000,-5.,5.);     hBottomMassDeltaY->setAxisTitle("M (GeV)");hBottomMassDeltaY->setAxisTitle("Y");

  hWplusPz       = dqm.book1dHisto("TTbar_WplusPz","W+ boson longitudinal momentum",200,-100.,100.);  hWplusPz->setAxisTitle("W+ boson longitudinal momentum");
  hWminPz        = dqm.book1dHisto("TTbar_WminPz","W- boson longitudinal momentum",200,-100.,100.);   hWminPz->setAxisTitle("W- boson longitudinal momentum");

}

