/*class DrellYanValidation
 *  
 *  Class to fill dqm monitor elements from existing EDM file
 *
 */
 
#include "Validation/EventGenerator/interface/DrellYanValidation.h"
#include "Validation/EventGenerator/interface/HepMCValidationHelper.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "TLorentzVector.h"

#include "CLHEP/Units/defs.h"
#include "CLHEP/Units/PhysicalConstants.h"

#include "DataFormats/Math/interface/LorentzVector.h"
#include "Validation/EventGenerator/interface/DQMHelper.h"
using namespace edm;

DrellYanValidation::DrellYanValidation(const edm::ParameterSet& iPSet): 
  wmanager_(iPSet,consumesCollector()),
  hepmcCollection_(iPSet.getParameter<edm::InputTag>("hepmcCollection")),
  _flavor(iPSet.getParameter<int>("decaysTo")),
  _name(iPSet.getParameter<std::string>("name")) 
{    
  hepmcCollectionToken_=consumes<HepMCProduct>(hepmcCollection_);
}

DrellYanValidation::~DrellYanValidation() {}

void DrellYanValidation::dqmBeginRun(const edm::Run& r, const edm::EventSetup& c) {
  c.getData( fPDGTable );
}

void DrellYanValidation::bookHistograms(DQMStore::IBooker &i, edm::Run const &, edm::EventSetup const &){

    ///Setting the DQM top directories
    std::string folderName = "Generator/DrellYan";
    folderName+=_name;
    DQMHelper dqm(&i); i.setCurrentFolder(folderName.c_str());
    
    // Number of analyzed events
    nEvt = dqm.book1dHisto("nEvt", "n analyzed Events", 1, 0., 1.,"bin","Number of Events");
    
    //Kinematics
    Zmass = dqm.book1dHisto("Zmass","inv. Mass Z", 70 ,0,140,"M_{Z} (GeV)","Number of Events");
    ZmassPeak = dqm.book1dHisto("ZmassPeak","inv. Mass Z", 80 ,80 ,100,"M_{Z} (GeV)","Number of Events");
    Zpt = dqm.book1dHisto("Zpt","Z pt",100,0,200,"P_{t}^{Z} (GeV)","Number of Events");
    ZptLog = dqm.book1dHisto("ZptLog","log(Z pt)",100,0.,5.,"log_{10}(P_{t}^{Z}) (log_{10}(GeV))","Number of Events");
    Zrap = dqm.book1dHisto("Zrap", "Z y", 100, -5, 5,"Y_{Z}","Number of Events");
    Zdaughters = dqm.book1dHisto("Zdaughters", "Z daughters", 60, -30, 30,"Z daughters (PDG ID)","Number of Events");

    dilep_mass = dqm.book1dHisto("dilep_mass","inv. Mass dilepton", 70 ,0,140,"M_{ll} (GeV)","Number of Events");
    dilep_massPeak = dqm.book1dHisto("dilep_massPeak","inv. Mass dilepton", 80 ,80 ,100,"M_{ll} (GeV)","Number of Events");
    dilep_pt = dqm.book1dHisto("dilep_pt","dilepton pt",100,0,200,"P_{t}^{ll} (GeV)","Number of Events");
    dilep_ptLog = dqm.book1dHisto("dilep_ptLog","log(dilepton pt)",100,0.,5.,"log_{10}(P_{t}^{ll}) (log_{10}(GeV))","Number of Events");
    dilep_rap = dqm.book1dHisto("dilep_rap", "dilepton y", 100, -5, 5,"Y_{ll}","Number of Events");

    gamma_energy = dqm.book1dHisto("gamma_energy", "photon energy in Z rest frame", 200, 0., 100.,"E_{#gamma}^{Z rest-frame} (GeV)","Number of Events");
    cos_theta_gamma_lepton = dqm.book1dHisto("cos_theta_gamma_lepton",      "cos_theta_gamma_lepton in Z rest frame",      200, -1, 1,"cos(#theta_{#gamma-lepton}^{Z rest-frame})","Number of Events");

    leadpt = dqm.book1dHisto("leadpt","leading lepton pt", 200, 0., 200.,"P_{t}^{1st-lepton}","Number of Events");    
    secpt  = dqm.book1dHisto("secpt","second lepton pt", 200, 0., 200.,"P_{t}^{2nd-lepton}","Number of Events");    
    leadeta = dqm.book1dHisto("leadeta","leading lepton eta", 100, -5., 5.,"#eta^{1st-lepton}","Number of Events");
    seceta  = dqm.book1dHisto("seceta","second lepton eta", 100, -5., 5.,"#eta^{2nd-lepton}","Number of Events");

  return;
}

void DrellYanValidation::analyze(const edm::Event& iEvent,const edm::EventSetup& iSetup)
{ 
  
  // we *DO NOT* rely on a Z entry in the particle listings!

  ///Gathering the HepMCProduct information
  edm::Handle<HepMCProduct> evt;
  iEvent.getByToken(hepmcCollectionToken_, evt);

  //Get EVENT
  const HepMC::GenEvent *myGenEvent = evt->GetEvent();

  double weight = wmanager_.weight(iEvent);

  //std::cout << "weight: " << weight << std::endl;

  nEvt->Fill(0.5,weight);

  std::vector<const HepMC::GenParticle*> allproducts; 

  //requires status 1 for leptons and neutrinos (except tau)
  int requiredstatus = (abs(_flavor) == 11 || abs(_flavor) == 12 || abs(_flavor) ==13 || abs(_flavor) ==14 || abs(_flavor) ==16) ? 1 : 3;

  bool vetotau = true; //(abs(_flavor) == 11 || abs(_flavor) == 12 || abs(_flavor) ==13 || abs(_flavor) ==14 || abs(_flavor) ==16) ? true : false;  

  for(HepMC::GenEvent::particle_const_iterator iter = myGenEvent->particles_begin(); iter != myGenEvent->particles_end(); ++iter) {
    if (vetotau) {
      if ((*iter)->status()==3 && abs((*iter)->pdg_id() == 15) ) return; 
    }
    if((*iter)->status()==requiredstatus) {
      if(abs((*iter)->pdg_id())==_flavor)
	allproducts.push_back(*iter);
    }
  }
 
  //nothing to do if we don't have 2 particles
  if (allproducts.size() < 2) return; 

  //sort them in pt
  std::sort(allproducts.begin(), allproducts.end(), HepMCValidationHelper::sortByPt); 

  //get the first element and the first following element with opposite charge 
  std::vector<const HepMC::GenParticle*> products;
  products.push_back(allproducts.front());
  const HepPDT::ParticleData* PData1 = fPDGTable->particle(HepPDT::ParticleID(allproducts.front()->pdg_id()));
  double charge1 = PData1->charge(); 
  for (unsigned int i = 1; i < allproducts.size(); ++i ){
    const HepPDT::ParticleData* PData2 = fPDGTable->particle(HepPDT::ParticleID(allproducts[i]->pdg_id()));
    double charge2 = PData2->charge();
    if (charge1*charge2 < 0) products.push_back(allproducts[i]);
  }

  //if we did not find any opposite charge pair there is nothing to do
  if (products.size() < 2) return; 

  assert(products[0]->momentum().perp() >= products[1]->momentum().perp()); 

  //leading lepton with pt > 20.
  if (products[0]->momentum().perp() < 20.) return;

  //assemble FourMomenta
  TLorentzVector lep1(products[0]->momentum().x(), products[0]->momentum().y(), products[0]->momentum().z(), products[0]->momentum().t()); 
  TLorentzVector lep2(products[1]->momentum().x(), products[1]->momentum().y(), products[1]->momentum().z(), products[1]->momentum().t()); 
  TLorentzVector dilepton_mom = lep1 + lep2;
  TLorentzVector dilepton_andphoton_mom = dilepton_mom;

  //mass > 60.
  if (dilepton_mom.M() < 60.) return;

  //find possible qed fsr photons
  std::vector<const HepMC::GenParticle*> fsrphotons;
  HepMCValidationHelper::findFSRPhotons(products, myGenEvent, 0.1, fsrphotons);
  
  Zdaughters->Fill(products[0]->pdg_id(),weight); 
  Zdaughters->Fill(products[1]->pdg_id(),weight); 

  std::vector<TLorentzVector> gammasMomenta;
  for (unsigned int ipho = 0; ipho < fsrphotons.size(); ++ipho){
    TLorentzVector phomom(fsrphotons[ipho]->momentum().x(), fsrphotons[ipho]->momentum().y(), fsrphotons[ipho]->momentum().z(), fsrphotons[ipho]->momentum().t()); 
    dilepton_andphoton_mom += phomom;
    Zdaughters->Fill(fsrphotons[ipho]->pdg_id(),weight);
    gammasMomenta.push_back(phomom);
  }  
  //Fill Z histograms
  Zmass->Fill(dilepton_andphoton_mom.M(),weight);
  ZmassPeak->Fill(dilepton_andphoton_mom.M(),weight);
  Zpt->Fill(dilepton_andphoton_mom.Pt(),weight);
  ZptLog->Fill(log10(dilepton_andphoton_mom.Pt()),weight); 
  Zrap->Fill(dilepton_andphoton_mom.Rapidity(),weight);

  //Fill dilepton histograms
  dilep_mass->Fill(dilepton_mom.M(),weight);
  dilep_massPeak->Fill(dilepton_mom.M(),weight);
  dilep_pt->Fill(dilepton_mom.Pt(),weight);
  dilep_ptLog->Fill(log10(dilepton_mom.Pt()),weight);
  dilep_rap->Fill(dilepton_mom.Rapidity(),weight); 

  //Fill lepton histograms 
  leadpt->Fill(lep1.Pt(),weight);
  secpt->Fill(lep2.Pt(),weight);
  leadeta->Fill(lep1.Eta(),weight);
  seceta->Fill(lep2.Eta(),weight);

  //boost everything in the Z frame
  TVector3 boost = dilepton_andphoton_mom.BoostVector();
  boost*=-1.;
  lep1.Boost(boost);
  lep2.Boost(boost);
  for (unsigned int ipho = 0; ipho < gammasMomenta.size(); ++ipho){
    gammasMomenta[ipho].Boost(boost);
  }
  std::sort(gammasMomenta.begin(), gammasMomenta.end(), HepMCValidationHelper::GreaterByE<TLorentzVector>);

  //fill gamma histograms
  if (gammasMomenta.size() != 0 && dilepton_andphoton_mom.M() > 50.) {
    gamma_energy->Fill(gammasMomenta.front().E(),weight);
    double dphi = lep1.DeltaR(gammasMomenta.front()) <  lep2.DeltaR(gammasMomenta.front()) ?
                  lep1.DeltaPhi(gammasMomenta.front()) : lep2.DeltaPhi(gammasMomenta.front());
    cos_theta_gamma_lepton->Fill(cos(dphi),weight);
  } 

}//analyze
