/*class WValidation
 *  
 *  Class to fill dqm monitor elements from existing EDM file
 *
 */
 
#include "Validation/EventGenerator/interface/WValidation.h"
#include "Validation/EventGenerator/interface/HepMCValidationHelper.h"
#include "TLorentzVector.h"

#include "CLHEP/Units/defs.h"
#include "CLHEP/Units/PhysicalConstants.h"

#include "DataFormats/Math/interface/LorentzVector.h"
#include "Validation/EventGenerator/interface/DQMHelper.h"
using namespace edm;

WValidation::WValidation(const edm::ParameterSet& iPSet): 
  wmanager_(iPSet,consumesCollector()),
  hepmcCollection_(iPSet.getParameter<edm::InputTag>("hepmcCollection")),
  _flavor(iPSet.getParameter<int>("decaysTo")),
  _name(iPSet.getParameter<std::string>("name")) 
{    

  hepmcCollectionToken_=consumes<HepMCProduct>(hepmcCollection_);
}

WValidation::~WValidation() {}

void WValidation::bookHistograms(DQMStore::IBooker &i, edm::Run const &, edm::EventSetup const &){
    ///Setting the DQM top directories
    std::string folderName = "Generator/W";
    folderName+=_name;
    DQMHelper dqm(&i); i.setCurrentFolder(folderName.c_str());
    
    // Number of analyzed events
    nEvt = dqm.book1dHisto("nEvt", "n analyzed Events", 1, 0., 1.,"bin","Number of Events");
    
    //Kinematics
    Wmass = dqm.book1dHisto("Wmass","inv. Mass W", 70 ,0,140,"M_{T}^{W} (GeV)","Number of Events");
    WmassPeak = dqm.book1dHisto("WmassPeak","inv. Mass W", 80 ,80 ,100,"M_{T}^{W} (GeV)","Number of Events");
    Wpt = dqm.book1dHisto("Wpt","W pt",100,0,200,"P_{T}^{W} (GeV)","Number of Events");
    WptLog = dqm.book1dHisto("WptLog","log(W pt)",100,0.,5.,"Log_{10}(P_{T}^{W}) (GeV)","Number of Events");
    Wrap = dqm.book1dHisto("Wrap", "W y", 100, -5, 5,"Y^{W}","Number of Events");
    Wdaughters = dqm.book1dHisto("Wdaughters", "W daughters", 60, -30, 30,"W daughters (PDG ID)","Number of Events");

    lepmet_mT = dqm.book1dHisto("lepmet_mT","lepton-met transverse mass", 70 ,0,140,"M_{T}^{Lepton_{T}+E_{T}^{Miss}} (GeV)","Number of Events");
    lepmet_mTPeak = dqm.book1dHisto("lepmet_mTPeak","lepton-met transverse mass", 80 ,80 ,100,"M_{T}^{Lepton_{T}+E_{T}^{Miss}} (GeV)","Number of Events");
    lepmet_pt = dqm.book1dHisto("lepmet_pt","lepton-met",100,0,200,"P_{T}^{Lepton_{T}+E_{T}^{Miss}} (GeV)","Number of Events");
    lepmet_ptLog = dqm.book1dHisto("lepmet_ptLog","log(lepton-met pt)",100,0.,5.,"log_{10}(P_{T}^{Lepton_{T}+E_{T}^{Miss}}) (Log_{10}(GeV))","Number of Events");

    gamma_energy = dqm.book1dHisto("gamma_energy", "photon energy in W rest frame", 200, 0., 100.,"E_{#gamma}^{W rest-frame}","Number of Events");
    cos_theta_gamma_lepton = dqm.book1dHisto("cos_theta_gamma_lepton",      "cos_theta_gamma_lepton in W rest frame",      200, -1, 1,"cos(#theta_{#gamma-lepton}^{W rest-frame})","Number of Events");

    leppt = dqm.book1dHisto("leadpt","lepton pt", 200, 0., 200.,"P_{t}^{Lead-Lepton} (GeV)","Number of Events");    
    met   = dqm.book1dHisto("met","met", 200, 0., 200.,"E_{T}^{Miss} (GeV)","Number of Events");    
    lepeta = dqm.book1dHisto("leadeta","leading lepton eta", 100, -5., 5.,"#eta^{Lead-Lepton}","Number of Events");

  return;
}

void WValidation::dqmBeginRun(const edm::Run& r, const edm::EventSetup& c) {
  c.getData( fPDGTable );
}

void WValidation::analyze(const edm::Event& iEvent,const edm::EventSetup& iSetup)
{ 
  
  // we *DO NOT* rely on a Z entry in the particle listings!

  ///Gathering the HepMCProduct information
  edm::Handle<HepMCProduct> evt;
  iEvent.getByToken(hepmcCollectionToken_, evt);

  //Get EVENT
  const HepMC::GenEvent *myGenEvent = evt->GetEvent();

  double weight =   wmanager_.weight(iEvent);

  nEvt->Fill(0.5,weight);

  std::vector<const HepMC::GenParticle*> allleptons; 
  std::vector<const HepMC::GenParticle*> allneutrinos; 

  //requires status 1 for leptons and neutrinos (except tau)
  int requiredstatus = (abs(_flavor) == 11 || abs(_flavor) ==13 ) ? 1 : 3;

  bool vetotau = true; //(abs(_flavor) == 11 || abs(_flavor) == 12 || abs(_flavor) ==13 || abs(_flavor) ==14 || abs(_flavor) ==16) ? true : false;  

  for(HepMC::GenEvent::particle_const_iterator iter = myGenEvent->particles_begin(); iter != myGenEvent->particles_end(); ++iter) {
    if (vetotau) {
      if ((*iter)->status()==3 && abs((*iter)->pdg_id() == 15) ) return;
    }
    if((*iter)->status()==requiredstatus) {
      //@todo: improve this selection	
      if((*iter)->pdg_id()==_flavor)
	allleptons.push_back(*iter);
      else if (abs((*iter)->pdg_id()) == abs(_flavor)+1)
	allneutrinos.push_back(*iter);	
    }
  }
 
  //nothing to do if we don't have 2 particles
  if (allleptons.empty() || allneutrinos.empty()) return;

  //sort them in pt
  std::sort(allleptons.begin(), allleptons.end(), HepMCValidationHelper::sortByPt); 
  std::sort(allneutrinos.begin(), allneutrinos.end(), HepMCValidationHelper::sortByPt); 

  //get the first lepton and the first neutrino, and check that one is particle one is antiparticle (product of pdgids < 0) 
  std::vector<const HepMC::GenParticle*> products;
  if (allleptons.front()->pdg_id() * allneutrinos.front()->pdg_id() > 0) return;	

  //require at least 20 GeV on the lepton
  if (allleptons.front()->momentum().perp() < 20. || allneutrinos.front()->momentum().perp() < 20. ) return;

  //find possible qed fsr photons
  std::vector<const HepMC::GenParticle*> selectedLepton;
  selectedLepton.push_back(allleptons.front());	
  std::vector<const HepMC::GenParticle*> fsrphotons;
  HepMCValidationHelper::findFSRPhotons(selectedLepton, myGenEvent, 0.1, fsrphotons);

  Wdaughters->Fill(allleptons.front()->pdg_id(),weight); 
  Wdaughters->Fill(allneutrinos.front()->pdg_id(),weight); 
 
  //assemble FourMomenta
  TLorentzVector lep1(allleptons[0]->momentum().x(), allleptons[0]->momentum().y(), allleptons[0]->momentum().z(), allleptons[0]->momentum().t()); 
  TLorentzVector lep2(allneutrinos[0]->momentum().x(), allneutrinos[0]->momentum().y(), allneutrinos[0]->momentum().z(), allneutrinos[0]->momentum().t()); 
  TLorentzVector dilepton_mom = lep1 + lep2;
  TLorentzVector dilepton_andphoton_mom = dilepton_mom;
  std::vector<TLorentzVector> gammasMomenta;
  for (unsigned int ipho = 0; ipho < fsrphotons.size(); ++ipho){
    TLorentzVector phomom(fsrphotons[ipho]->momentum().x(), fsrphotons[ipho]->momentum().y(), fsrphotons[ipho]->momentum().z(), fsrphotons[ipho]->momentum().t()); 
    dilepton_andphoton_mom += phomom;
    Wdaughters->Fill(fsrphotons[ipho]->pdg_id(),weight);
    gammasMomenta.push_back(phomom);
  }  
  //Fill "true" W histograms
  Wmass->Fill(dilepton_andphoton_mom.M(),weight);
  WmassPeak->Fill(dilepton_andphoton_mom.M(),weight);
  Wpt->Fill(dilepton_andphoton_mom.Pt(),weight);
  WptLog->Fill(log10(dilepton_andphoton_mom.Pt()),weight); 
  Wrap->Fill(dilepton_andphoton_mom.Rapidity(),weight);

  TLorentzVector met_mom = HepMCValidationHelper::genMet(myGenEvent, -3., 3.);
  TLorentzVector lep1T(lep1.Px(), lep1.Py(), 0., lep1.Et());
  TLorentzVector lepmet_mom = lep1T + met_mom;
  //Fill lepmet histograms
  lepmet_mT->Fill(lepmet_mom.M(),weight);
  lepmet_mTPeak->Fill(lepmet_mom.M(),weight);
  lepmet_pt->Fill(lepmet_mom.Pt(),weight);
  lepmet_ptLog->Fill(log10(lepmet_mom.Pt()),weight);

  //Fill lepton histograms 
  leppt->Fill(lep1.Pt(),weight);
  lepeta->Fill(lep1.Eta(),weight);
  met->Fill(met_mom.Pt(),weight);	

  //boost everything in the W frame
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
    double dphi = lep1.DeltaR(gammasMomenta.front());
    cos_theta_gamma_lepton->Fill(cos(dphi),weight);
  } 


}//analyze
