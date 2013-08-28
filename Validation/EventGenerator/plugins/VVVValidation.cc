
#include "Validation/EventGenerator/interface/VVVValidation.h"
#include "Validation/EventGenerator/interface/HepMCValidationHelper.h"


#include "CLHEP/Units/defs.h"
#include "CLHEP/Units/PhysicalConstants.h"


using namespace edm;
using namespace std;

VVVValidation::VVVValidation(const edm::ParameterSet& iPSet): 
  _wmanager(iPSet),
  hepmcCollection_(iPSet.getParameter<edm::InputTag>("hepmcCollection")),
  genparticleCollection_(iPSet.getParameter<edm::InputTag>("genparticleCollection")),
  genjetCollection_(iPSet.getParameter<edm::InputTag>("genjetsCollection")),
  matchPr_(iPSet.getParameter<double>("matchingPrecision")),
  _lepStatus(iPSet.getParameter<double>("lepStatus")),
  verbosity_(iPSet.getUntrackedParameter<unsigned int>("verbosity",0))
{    
  dbe = 0;
  dbe = edm::Service<DQMStore>().operator->();
}

VVVValidation::~VVVValidation() {}

void VVVValidation::beginJob()
{



  if(dbe){

    dbe->setCurrentFolder("Generator/VVVJets");
	
    
    TH1::SetDefaultSumw2();
    nEvt = dbe->book1D("nEvt", "n analyzed Events", 1, 0., 1.);

    genJetMult = dbe->book1D("genJetMult", "GenJet multiplicity", 50, 0, 50);
    genJetEnergy = dbe->book1D("genJetEnergy", "Log10(GenJet energy)", 60, -1, 5);
    genJetPt = dbe->book1D("genJetPt", "Log10(GenJet pt)", 60, -1, 5);
    genJetEta = dbe->book1D("genJetEta", "GenJet eta", 220, -11, 11);
    genJetPhi = dbe->book1D("genJetPhi", "GenJet phi", 360, -180, 180);
    genJetDeltaEtaMin = dbe->book1D("genJetDeltaEtaMin", "GenJet minimum rapidity gap", 30, 0, 30);
    
    genJetPto1 = dbe->book1D("genJetPto1", "GenJet multiplicity above 1 GeV", 50, 0, 50);
    genJetPto30 = dbe->book1D("genJetPto30", "GenJet multiplicity above 30 GeV", 10, -0.5, 9.5);
    genJetPto50 = dbe->book1D("genJetPto50", "GenJet multiplicity above 50 GeV", 10, -0.5, 9.5);
    genJetPto100 = dbe->book1D("genJetPto100", "GenJet multiplicity above 100 GeV", 10, -0.5, 9.5);
    genJetCentral = dbe->book1D("genJetCentral", "GenJet multiplicity |eta|.lt.2.5", 10, -0.5, 9.5);

    WW_TwoJEt_JetsM= dbe->book1D("WW_TwoJEt_JetsM", "2 jet invariant mass in WW 2 J events", 200, 0, 2000);
    h_l_jet_eta = dbe->book1D("h_l_jet_eta", "leading jet eta",  30, -5.2, 5.2);
    h_l_jet_pt = dbe->book1D("h_l_jet_pt", "leading jet pt", 50, 0, 300);
    h_sl_jet_eta = dbe->book1D("h_sl_jet_eta", "leading jet eta",  30, -5.2, 5.2);
    h_sl_jet_pt = dbe->book1D("h_sl_jet_pt", "leading jet pt", 50, 0, 300);
    h_ssl_jet_eta = dbe->book1D("h_ssl_jet_eta", "leading jet eta",  30, -5.2, 5.2); 
    h_ssl_jet_pt = dbe->book1D("h_ssl_jet_pt", "leading jet pt", 50, 0, 300);


    genJetTotPt = dbe->book1D("genJetTotPt", "Log10(GenJet total pt)", 100, 0, 1000);
 
    h_dr = dbe->book1D("h_dr", "DeltaR between leptons and jets", 30, 0, 2);
    h_mWplus = dbe->book1D("h_mWplus", "M W^{+}", 50, 0, 200);
    h_phiWplus= dbe->book1D("h_phiWplus", "#phi W^{+}", 30, -3.5, 3.5);
    h_ptWplus = dbe->book1D("h_ptWplus", "P_{T} W^{+}", 50, 0, 200);
    h_yWplus = dbe->book1D("h_yWplus", "Y W^{+}", 30, -3.5, 3.5);

    h_mWminus= dbe->book1D("h_mWminus", "M W^{-}", 50, 0, 200);
    h_phiWminus= dbe->book1D("h_phiWminus", "#phi W^{-}", 30, -3.5, 3.5);
    h_ptWminus= dbe->book1D("h_ptWminus", "P_{T} W^{-}", 50, 0, 200);
    h_yWminus= dbe->book1D("h_yWminus", "Y W^{-}", 30, -3.5, 3.5);

    h_mZ= dbe->book1D("h_mZ", "M Z", 50, 0, 200);
    h_phiZ= dbe->book1D("h_phiZ", "#phi Z", 30, -3.5, 3.5);
    h_ptZ= dbe->book1D("h_ptZ", "P_{T} Z", 50, 0, 200);
    h_yZ= dbe->book1D("h_yZ", "Y Z", 30, -3.5, 3.5);

    h_mWplus_3b = dbe->book1D("h_mWplus_3b", "M W^{+}", 50, 0, 200);
    h_phiWplus_3b= dbe->book1D("h_phiWplus_3b", "#phi W^{+}", 30, -3.5, 3.5);
    h_ptWplus_3b = dbe->book1D("h_ptWplus_3b", "P_{T} W^{+}", 50, 0, 200);
    h_yWplus_3b = dbe->book1D("h_yWplus_3b", "Y W^{+}", 30, -3.5, 3.5);

    h_mWminus_3b= dbe->book1D("h_mWminus_3b", "M W^{-}", 50, 0, 200);
    h_phiWminus_3b= dbe->book1D("h_phiWminus_3b", "#phi W^{-}", 30, -3.5, 3.5);
    h_ptWminus_3b= dbe->book1D("h_ptWminus_3b", "P_{T} W^{-}", 50, 0, 200);
    h_yWminus_3b= dbe->book1D("h_yWminus_3b", "Y W^{-}", 30, -3.5, 3.5);

    h_mZ_3b= dbe->book1D("h_mZ_3b", "M Z", 50, 0, 200);
    h_phiZ_3b= dbe->book1D("h_phiZ_3b", "#phi Z", 30, -3.5, 3.5);
    h_ptZ_3b= dbe->book1D("h_ptZ_3b", "P_{T} Z", 50, 0, 200);
    h_yZ_3b= dbe->book1D("h_yZ_3b", "Y Z", 30, -3.5, 3.5);

    h_mWW= dbe->book1D("h_mWW", "M W^{-}W^{+}", 200, 0, 2000);
    h_phiWW= dbe->book1D("h_phiWW", "#phi W^{-}W^{+}", 30, -3.5, 3.5);
    h_ptWW= dbe->book1D("h_ptWW", "P_{T} W^{-}W^{+}", 50, 0, 2000);
    h_yWW =dbe->book1D("h_yWW", "Y W^{-}W^{+}", 30, -3.5, 3.5);

    h_mWZ= dbe->book1D("h_mWZ", "M W Z", 200, 0, 2000);
    h_phiWZ= dbe->book1D("h_phiWZ", "#phi W Z", 30, -3.5, 3.5);
    h_ptWZ= dbe->book1D("h_ptWZ", "P_{T} W Z", 50, 0, 2000);
    h_yWZ =dbe->book1D("h_yWZ", "Y W Z", 30, -3.5, 3.5);

    h_mZZ= dbe->book1D("h_mZZ", "M Z Z", 200, 0, 2000);
    h_phiZZ= dbe->book1D("h_phiZZ", "#phi Z Z", 30, -3.5, 3.5);
    h_ptZZ= dbe->book1D("h_ptZZ", "P_{T} Z Z", 50, 0, 2000);
    h_yZZ =dbe->book1D("h_yZZ", "Y Z Z", 30, -3.5, 3.5);

    h_mWWW= dbe->book1D("h_mWWW", "M W W W ", 200, 0, 2000);
    h_phiWWW= dbe->book1D("h_phiWWW", "#phi W W W", 30, -3.5, 3.5);
    h_ptWWW= dbe->book1D("h_ptWWW", "P_{T} W W W", 50, 0, 2000);
    h_yWWW =dbe->book1D("h_yWWW", "Y W W W", 30, -3.5, 3.5);

    h_mWWZ= dbe->book1D("h_mWWZ", "M W W Z ", 200, 0, 2000);
    h_phiWWZ= dbe->book1D("h_phiWWZ", "#phi W W Z", 30, -3.5, 3.5);
    h_ptWWZ= dbe->book1D("h_ptWWZ", "P_{T} W W Z", 50, 0, 2000);
    h_yWWZ =dbe->book1D("h_yWWZ", "Y W W Z", 30, -3.5, 3.5);

    h_mWZZ= dbe->book1D("h_mWZZ", "M W Z Z ", 200, 0, 2000);
    h_phiWZZ= dbe->book1D("h_phiWZZ", "#phi W Z Z", 30, -3.5, 3.5);
    h_ptWZZ= dbe->book1D("h_ptWZZ", "P_{T} W Z Z", 50, 0, 2000);
    h_yWZZ =dbe->book1D("h_yWZZ", "Y W Z Z", 30, -3.5, 3.5);

    h_mZZZ= dbe->book1D("h_mZZZ", "M Z Z Z ", 200, 0, 2000);
    h_phiZZZ= dbe->book1D("h_phiZZZ", "#phi Z Z Z", 30, -3.5, 3.5);
    h_ptZZZ= dbe->book1D("h_ptZZZ", "P_{T} Z Z Z", 50, 0, 2000);
    h_yZZZ =dbe->book1D("h_yZZZ", "Y Z Z Z", 30, -3.5, 3.5);

 
	  
    leading_l_pt = dbe->book1D("leading_l_pt", "leading lepton pt", 50, 0, 200);
    subleading_l_pt = dbe->book1D("subleading_l_pt", "subleading lepton pt", 50, 0, 200);
    subsubleading_l_pt = dbe->book1D("subsubleading_l_pt", "subsubleading lepton pt", 50, 0, 200);
    leading_l_eta = dbe->book1D("leading_l_eta", "leading lepton eta",  30, -3.5, 3.5);
    subleading_l_eta = dbe->book1D("subleading_l_eta", "subleading lepton eta",  30, -3.5, 3.5);
    subsubleading_l_eta = dbe->book1D("subsubleading_l_eta", "subsubleading lepton eta",  30, -3.5, 3.5);
    mll= dbe->book1D("mll", "ll mass (all combinations)", 50, 0, 200);
    ptll= dbe->book1D("ptll", "ll Transverse Momentum (all combinations)", 50, 0, 200);
    mlll= dbe->book1D("mlll", "lll mass ", 50, 0, 200);
    ptlll= dbe->book1D("ptlll", "lll Transverse Momentum ", 50, 0, 2000);
    mlllnununu= dbe->book1D("mlllnununu", "lll nununu mass ", 50, 0, 2000);
    mtlllnununu= dbe->book1D("mtlllnununu", "lll nununu transverse mass ", 50, 0, 2000);
    ptlllnununu= dbe->book1D("ptlllnununu", "lll nununu Transverse Momentum ", 50, 0, 2000);
	  

  }
  return;
}

void VVVValidation::endJob(){return;}
void VVVValidation::beginRun(const edm::Run& iRun,const edm::EventSetup& iSetup)
{
  iSetup.getData( fPDGTable );
  return;
}
void VVVValidation::endRun(const edm::Run& iRun,const edm::EventSetup& iSetup){return;}
void VVVValidation::analyze(const edm::Event& iEvent,const edm::EventSetup& iSetup)
{ 

  unsigned int initSize = 1000;

  edm::Handle<HepMCProduct> evt;
  iEvent.getByLabel(hepmcCollection_, evt);

  HepMC::GenEvent *myGenEvent = new HepMC::GenEvent(*(evt->GetEvent()));

  double weight = _wmanager.weight(iEvent);

  nEvt->Fill(0.5, weight);

  int st_3_leptons=0;
  int n_particles=0;
  int Wpst3=0;
  int Wmst3=0;
  int taust3=0;

  std::vector<const HepMC::GenParticle*> mothercoll;
  std::vector<const HepMC::GenParticle*> GenLeptons;
  std::vector<const HepMC::GenParticle*> GenNeutrinos;
  mothercoll.reserve(initSize);
  GenLeptons.reserve(initSize);
  GenNeutrinos.reserve(initSize);
  for (HepMC::GenEvent::particle_const_iterator iter = myGenEvent->particles_begin(); iter != myGenEvent->particles_end(); ++iter){
    double id = (*iter)->pdg_id();

    if((*iter)->status()==3&&(fabs(id)==11 ||fabs(id)==13) && (*iter)->momentum().perp()>20.&& fabs((*iter)->momentum().eta())<2.5){
      st_3_leptons++;
    }
    if((*iter)->status()==1&&(fabs(id)==16))taust3++;
    if((*iter)->status()==3&&(id==24))Wpst3++;
    if((*iter)->status()==3&&(id==-24))Wmst3++;
    if((*iter)->status()==3&&(fabs(id)==24 || fabs(id)==23)){
     mothercoll.push_back(*iter);
    }

    if((*iter)->status()==_lepStatus && (*iter)->momentum().perp()>20.&& fabs((*iter)->momentum().eta())<2.5 ){
      n_particles ++;
      if (fabs(id)==11 ||fabs(id)==13){
        GenLeptons.push_back(*iter);
      }
      if (fabs(id)==12 ||fabs(id)==14){
        GenNeutrinos.push_back(*iter);
      }
    }
  }

  vector<TLorentzVector> wpluss;
  vector<TLorentzVector> wmins;
  vector<TLorentzVector> z;
  wpluss.clear();
  wmins.clear();
  z.clear();
  int Wmin=0;
  int Wpl=0;
  int Z=0;
  int l_bcode[3];
  int nu_bcode[3];


  if(taust3>0)return;
  
  for(unsigned int i = 0 ; i< mothercoll.size();i++){
    double id = mothercoll[i]->pdg_id();

    ////////////////////
    //&&&&W bosons&&&&//
    ////////////////////
    if(fabs(id)==24){
      double dr_min=999.;
      double dr=999.;
      for(unsigned int k=0 ; k< GenLeptons.size() ;k++){
        for(unsigned int kl=0 ; kl< GenNeutrinos.size() ;kl++){
          double lepton_px=GenLeptons[k]->momentum().px();
          double lepton_py=GenLeptons[k]->momentum().py();
          double lepton_pz=GenLeptons[k]->momentum().pz();
          double lepton_e=GenLeptons[k]->momentum().e();
          TLorentzVector l;
          l.SetPxPyPzE(lepton_px,lepton_py,lepton_pz,lepton_e); 
          double nu_px=GenNeutrinos[kl]->momentum().px();
          double nu_py=GenNeutrinos[kl]->momentum().py();
          double nu_pz=GenNeutrinos[kl]->momentum().pz();
          double nu_e=GenNeutrinos[kl]->momentum().e();
          TLorentzVector nu;
          nu.SetPxPyPzE(nu_px,nu_py,nu_pz,nu_e);
          double l_id= GenLeptons[k]->pdg_id();  
          double nu_id= GenNeutrinos[kl]->pdg_id();
          dr= deltaR((l+nu).PseudoRapidity(),(l+nu).Phi(),mothercoll[i]->momentum().eta(),mothercoll[i]->momentum().phi());
          if((id*l_id)<0 &&(l_id*nu_id)<0 &&( fabs(nu_id)== (fabs(l_id)+1) )&& dr<0.5/*&& (l+nu).M()>6.*/){
            if(dr<dr_min)dr_min=dr;
            if(dr>dr_min)continue;
          }
        }
      }
      for(unsigned int k=0 ; k< GenLeptons.size() ;k++){
        for(unsigned int kl=0 ; kl< GenNeutrinos.size() ;kl++){
          double lepton_px=GenLeptons[k]->momentum().px();
          double lepton_py=GenLeptons[k]->momentum().py();
          double lepton_pz=GenLeptons[k]->momentum().pz();
          double lepton_e=GenLeptons[k]->momentum().e();
          TLorentzVector l;
          l.SetPxPyPzE(lepton_px,lepton_py,lepton_pz,lepton_e); 
          double nu_px=GenNeutrinos[kl]->momentum().px();
          double nu_py=GenNeutrinos[kl]->momentum().py();
          double nu_pz=GenNeutrinos[kl]->momentum().pz();
          double nu_e=GenNeutrinos[kl]->momentum().e();
          TLorentzVector nu;
          nu.SetPxPyPzE(nu_px,nu_py,nu_pz,nu_e);
          double l_id= GenLeptons[k]->pdg_id();  
          double nu_id= GenNeutrinos[kl]->pdg_id();
          double der= deltaR((l+nu).PseudoRapidity(),(l+nu).Phi(),mothercoll[i]->momentum().eta(),mothercoll[i]->momentum().phi());
          if((id*l_id)<0 && (l_id*nu_id)<0 &&( fabs(nu_id)== (fabs(l_id)+1) )&& der==dr_min){
            l_bcode[i]=GenLeptons[k]->barcode();
            nu_bcode[i]=GenNeutrinos[kl]->barcode();
            if((i==0)|| (i==1 &&  (l_bcode[i]!=l_bcode[i-1]) && (nu_bcode[i]!=nu_bcode[i-1])  )||
               (i==2 &&  (l_bcode[i]!=l_bcode[i-1]) && (nu_bcode[i]!=nu_bcode[i-1]) && (l_bcode[i]!=l_bcode[i-2]) && (nu_bcode[i]!=nu_bcode[i-2]) )
              ){
              if(id==24){
                Wpl++;
                wpluss.push_back((l+nu));
                h_mWplus->Fill((l+nu).M(),weight);
                h_phiWplus->Fill((l+nu).Phi(),weight);
                h_ptWplus->Fill((l+nu).Pt(),weight);
                h_yWplus->Fill((l+nu).Rapidity(),weight);
              }
              if(id==-24){
                Wmin++;
                wmins.push_back((l+nu));
                h_mWminus->Fill((l+nu).M(),weight);
                h_phiWminus->Fill((l+nu).Phi(),weight);
                h_ptWminus->Fill((l+nu).Pt(),weight);
                h_yWminus->Fill((l+nu).Rapidity(),weight);
              }
            } 
          }
        }
      }
    }
    
    ////////////////////
    //&&&&Z bosons&&&&//
    ////////////////////
    if(fabs(id)==23){
      double dr_min=999.;
      double dr=999.;
      for(unsigned int k=0 ; k< GenLeptons.size() ;k++){
        for(unsigned int kl=k ; kl< GenLeptons.size() ;kl++){
          if(k==kl)continue;
          double lepton_px=GenLeptons[k]->momentum().px();
          double lepton_py=GenLeptons[k]->momentum().py();
          double lepton_pz=GenLeptons[k]->momentum().pz();
          double lepton_e=GenLeptons[k]->momentum().e();
          TLorentzVector l;
          l.SetPxPyPzE(lepton_px,lepton_py,lepton_pz,lepton_e); 
          double nu_px=GenLeptons[kl]->momentum().px();
          double nu_py=GenLeptons[kl]->momentum().py();
          double nu_pz=GenLeptons[kl]->momentum().pz();
          double nu_e=GenLeptons[kl]->momentum().e();
          TLorentzVector nu;
          nu.SetPxPyPzE(nu_px,nu_py,nu_pz,nu_e);
          double l_id= GenLeptons[k]->pdg_id();  
          double nu_id= GenLeptons[kl]->pdg_id();
          dr= deltaR((l+nu).PseudoRapidity(),(l+nu).Phi(),mothercoll[i]->momentum().eta(),mothercoll[i]->momentum().phi());
          if((l_id*nu_id)<0 &&( fabs(nu_id)== (fabs(l_id)) && dr<0.5)/*&& (l+nu).M()>6.*/){
            if(dr<dr_min)dr_min=dr;
            if(dr>dr_min)continue;
          }
        }
      }
      for(unsigned int k=0 ; k< GenLeptons.size() ;k++){
        for(unsigned int kl=k ; kl< GenLeptons.size() ;kl++){
          if(k==kl)continue;
          double lepton_px=GenLeptons[k]->momentum().px();
          double lepton_py=GenLeptons[k]->momentum().py();
          double lepton_pz=GenLeptons[k]->momentum().pz();
          double lepton_e=GenLeptons[k]->momentum().e();
          TLorentzVector l;
          l.SetPxPyPzE(lepton_px,lepton_py,lepton_pz,lepton_e); 
          double nu_px=GenLeptons[kl]->momentum().px();
          double nu_py=GenLeptons[kl]->momentum().py();
          double nu_pz=GenLeptons[kl]->momentum().pz();
          double nu_e=GenLeptons[kl]->momentum().e();
          TLorentzVector nu;
          nu.SetPxPyPzE(nu_px,nu_py,nu_pz,nu_e);
          double l_id= GenLeptons[k]->pdg_id();  
          double nu_id= GenLeptons[kl]->pdg_id();
          double der= deltaR((l+nu).PseudoRapidity(),(l+nu).Phi(),mothercoll[i]->momentum().eta(),mothercoll[i]->momentum().phi());
          if((l_id*nu_id)<0 &&( fabs(nu_id)== (fabs(l_id)) )&& der==dr_min ){
            l_bcode[i]=GenLeptons[k]->barcode();
            nu_bcode[i]=GenLeptons[kl]->barcode();
           if((i==0)|| (i==1 &&  (l_bcode[i]!=l_bcode[i-1]) && (nu_bcode[i]!=nu_bcode[i-1])  )||
               (i==2 &&  (l_bcode[i]!=l_bcode[i-1]) && (nu_bcode[i]!=nu_bcode[i-1]) && (l_bcode[i]!=l_bcode[i-2]) && (nu_bcode[i]!=nu_bcode[i-2]) )
             ){
              Z++;
              z.push_back((l+nu));
              h_mZ->Fill((l+nu).M(),weight);
              h_phiZ->Fill((l+nu).Phi(),weight);
              h_ptZ->Fill((l+nu).Pt(),weight);
              h_yZ->Fill((l+nu).Rapidity(),weight); 
            }          
          }
        }
      }
    }
  }


  if((Wmin+Wpl)>3) cout<<"3ten fazla W adayı?!?"<<endl;

  if( ((Wmin+Wpl)==3) || ((Wmin+Wpl)==2 && Z==1) || (( Wmin+Wpl )==1 && Z==2) || (Z==3) ){


    for(unsigned int i=0; i<wmins.size();i++){
      h_mWminus_3b->Fill(wmins[i].M(),weight);
      h_phiWminus_3b->Fill(wmins[i].Phi(),weight);
      h_ptWminus_3b->Fill(wmins[i].Pt(),weight);
      h_yWminus_3b->Fill(wmins[i].Rapidity(),weight);
    }
    for(unsigned int i=0; i<wpluss.size();i++){
      h_mWplus_3b->Fill(wpluss[i].M(),weight);
      h_phiWplus_3b->Fill(wpluss[i].Phi(),weight);
      h_ptWplus_3b->Fill(wpluss[i].Pt(),weight);
      h_yWplus_3b->Fill(wpluss[i].Rapidity(),weight);
    }
    for(unsigned int i=0; i<z.size();i++){
      h_mZ_3b->Fill(z[i].M(),weight);
      h_phiZ_3b->Fill(z[i].Phi(),weight);
      h_ptZ_3b->Fill(z[i].Pt(),weight);
      h_yZ_3b->Fill(z[i].Rapidity(),weight);
    }
 
    if((Wmin+Wpl)==3){
        TLorentzVector W1;
        TLorentzVector W2;
        TLorentzVector W3;
      if(Wmin==2 && Wpl==1){
        W1=wmins[0];
        W2=wmins[1];
        W3=wpluss[0];
      }
      if(Wmin==1 && Wpl==2){
        W1=wmins[0];
        W2=wpluss[0];
        W3=wpluss[1];
      }
      if(W1.M()<10. ||W2.M()<10.||W3.M()<10.)cout<<taust3<<endl;
      h_mWWW->Fill((W1+W2+W3).M(),weight);
      h_phiWWW->Fill((W1+W2+W3).Phi(),weight);
      h_ptWWW->Fill((W1+W2+W3).Pt(),weight);
      h_yWWW->Fill((W1+W2+W3).Rapidity(),weight);

      h_mWW->Fill((W1+W2).M(),weight);
      h_mWW->Fill((W1+W3).M(),weight);
      h_mWW->Fill((W2+W3).M(),weight);

      h_phiWW->Fill((W1+W2).Phi(),weight);
      h_phiWW->Fill((W1+W3).Phi(),weight);
      h_phiWW->Fill((W2+W3).Phi(),weight);

      h_ptWW->Fill((W1+W2).Pt(),weight);
      h_ptWW->Fill((W1+W3).Pt(),weight);
      h_ptWW->Fill((W2+W3).Pt(),weight);

      h_yWW->Fill((W1+W2).Rapidity(),weight);
      h_yWW->Fill((W1+W3).Rapidity(),weight);
      h_yWW->Fill((W2+W3).Rapidity(),weight);


    }
    if((Wmin+Wpl)==2 && Z==1){
      TLorentzVector W1;
      TLorentzVector W2;
      TLorentzVector Z1;
      Z1=z[0];
      if(Wmin==1 &&Wpl==1){
        W1=wmins[0];
        W2=wpluss[0];
      }  
      if(Wmin==2 &&Wpl==0){
        W1=wmins[0];
        W2=wmins[1];
      }  
      if(Wmin==0 &&Wpl==2){
        W1=wpluss[0];
        W2=wpluss[1];
      }
      h_mWW->Fill((W1+W2).M(),weight);
      h_phiWW->Fill((W1+W2).Phi(),weight);
      h_ptWW->Fill((W1+W2).Pt(),weight);
      h_yWW->Fill((W1+W2).Rapidity(),weight);

      h_mWZ->Fill((W1+Z1).M(),weight);
      h_mWZ->Fill((W2+Z1).M(),weight);

      h_phiWZ->Fill((W1+Z1).Phi(),weight);
      h_phiWZ->Fill((W2+Z1).Phi(),weight);

      h_ptWZ->Fill((W1+Z1).Pt(),weight);
      h_ptWZ->Fill((W2+Z1).Pt(),weight);

      h_yWZ->Fill((W1+Z1).Rapidity(),weight);
      h_yWZ->Fill((W2+Z1).Rapidity(),weight);

      h_mWWZ->Fill((W1+W2+Z1).M(),weight);
      h_phiWWZ->Fill((W1+W2+Z1).Phi(),weight);

      h_ptWWZ->Fill((W1+W2+Z1).Pt(),weight);
      h_yWWZ->Fill((W1+W2+Z1).Rapidity(),weight);

  		  
    }

    if((Wmin+Wpl)==1 && Z==2){
      TLorentzVector Z1;
      TLorentzVector Z2;
      TLorentzVector W1;
      Z1=z[0];
      Z2=z[1];
      if(Wmin==1){
        W1=wmins[0];
      }  
      if(Wpl==1){
        W1=wpluss[0];
      }  
      h_mZZ->Fill((Z1+Z2).M(),weight);
      h_phiZZ->Fill((Z1+Z2).Phi(),weight);
      h_ptZZ->Fill((Z1+Z2).Pt(),weight);
      h_yZZ->Fill((Z1+Z2).Rapidity(),weight);

      h_mWZ->Fill((W1+Z1).M(),weight);
      h_mWZ->Fill((W1+Z2).M(),weight);

      h_phiWZ->Fill((W1+Z1).Phi(),weight);
      h_phiWZ->Fill((W1+Z2).Phi(),weight);

      h_ptWZ->Fill((W1+Z1).Pt(),weight);
      h_ptWZ->Fill((W1+Z2).Pt(),weight);

      h_yWZ->Fill((W1+Z1).Rapidity(),weight);
      h_yWZ->Fill((W1+Z2).Rapidity(),weight);

      h_mWZZ->Fill((Z1+Z2+W1).M(),weight);
      h_phiWZZ->Fill((Z1+Z2+W1).Phi(),weight);
      h_ptWZZ->Fill((Z1+Z2+W1).Pt(),weight);
      h_yWZZ->Fill((Z1+Z2+W1).Rapidity(),weight);

  		  
    }

    if(Z==3){
      TLorentzVector Z1;
      TLorentzVector Z2;
      TLorentzVector Z3;

      Z1=z[0];
      Z2=z[1];
      Z3=z[2];

      if(Z1.M()<10. ||Z2.M()<10.||Z3.M()<10.)cout<<taust3<<endl;

      h_mZZZ->Fill((Z1+Z2+Z3).M(),weight);
      h_phiZZZ->Fill((Z1+Z2+Z3).Phi(),weight);
      h_ptZZZ->Fill((Z1+Z2+Z3).Pt(),weight);
      h_yZZZ->Fill((Z1+Z2+Z3).Rapidity(),weight);

      h_mZZ->Fill((Z1+Z2).M(),weight);
      h_mZZ->Fill((Z1+Z3).M(),weight);
      h_mZZ->Fill((Z2+Z3).M(),weight);

      h_phiZZ->Fill((Z1+Z2).Phi(),weight);
      h_phiZZ->Fill((Z1+Z3).Phi(),weight);
      h_phiZZ->Fill((Z2+Z3).Phi(),weight);

      h_ptZZ->Fill((Z1+Z2).Pt(),weight);
      h_ptZZ->Fill((Z1+Z3).Pt(),weight);
      h_ptZZ->Fill((Z2+Z3).Pt(),weight);

      h_yZZ->Fill((Z1+Z2).Rapidity(),weight);
      h_yZZ->Fill((Z1+Z3).Rapidity(),weight);
      h_yZZ->Fill((Z2+Z3).Rapidity(),weight);

    }


	  
    edm::Handle<reco::GenJetCollection> genJets;
    iEvent.getByLabel(genjetCollection_, genJets );

    std::vector<const reco::GenJet*> selected_jets;
    selected_jets.reserve(initSize);

    int nJets = 0;
    int nJetso1 = 0;
    int nJetso30 = 0;
    int nJetso50 = 0;
    int nJetso100 = 0;
    int nJetsCentral = 0;
    double totPt = 0.;
    double dr=99.;
    std::vector<double> jetEta;
    jetEta.reserve(initSize);
    for (reco::GenJetCollection::const_iterator iter=genJets->begin();iter!=genJets->end();++iter){
      dr=0;
      bool matched_lepton=false;
      for(unsigned int i=0 ; i< GenLeptons.size() ;i++){
        dr= deltaR(GenLeptons[i]->momentum().eta(),GenLeptons[i]->momentum().phi(),(*iter).eta(),(*iter).phi());
        h_dr->Fill(dr,weight);
        if(dr<0.5)matched_lepton=true;
      }
      if(matched_lepton)continue;
      selected_jets.push_back(&*iter);
      nJets++;
      double pt = (*iter).pt();
      totPt += pt;
      if (pt > 1.) nJetso1++;
      double eta = (*iter).eta();
      if (std::fabs(eta) < 5.&&pt > 30.) nJetso30++;
      if (std::fabs(eta) < 5.&&pt > 50.) nJetso50++;
      if (std::fabs(eta) < 5.&&pt > 100.) nJetso100++;
      if ( std::fabs(eta) < 2.5 ) nJetsCentral++;
      jetEta.push_back(eta);

      genJetEnergy->Fill(std::log10((*iter).energy()),weight);
      genJetPt->Fill(std::log10(pt),weight);
      genJetEta->Fill(eta,weight);
      genJetPhi->Fill((*iter).phi()/CLHEP::degree,weight);
    }
    if(nJetso30==2){
 
      TLorentzVector j1;
      TLorentzVector j2;
      j1.SetPtEtaPhiE(selected_jets[0]->pt(),selected_jets[0]->eta(),selected_jets[0]->phi(),selected_jets[0]->energy());
      j2.SetPtEtaPhiE(selected_jets[1]->pt(),selected_jets[1]->eta(),selected_jets[1]->phi(),selected_jets[1]->energy());
      WW_TwoJEt_JetsM->Fill((j1+j2).M(),weight);
    } 
    if(nJetso30>0)h_l_jet_eta->Fill(selected_jets[0]->eta());
    if(nJetso30>0)h_l_jet_pt->Fill(selected_jets[0]->pt());
    if(nJetso30>1)h_sl_jet_eta->Fill(selected_jets[1]->eta());
    if(nJetso30>1)h_sl_jet_pt->Fill(selected_jets[1]->pt());
    if(nJetso30>2)h_ssl_jet_eta->Fill(selected_jets[2]->eta());
    if(nJetso30>2)h_ssl_jet_pt->Fill(selected_jets[2]->pt());

    genJetMult->Fill(nJets,weight);
    genJetPto1->Fill(nJetso1,weight);
    genJetPto30->Fill(nJetso30,weight);
    genJetPto50->Fill(nJetso50,weight);
    genJetPto100->Fill(nJetso100,weight);
    genJetCentral->Fill(nJetsCentral,weight);

    genJetTotPt->Fill(totPt,weight);

    double deltaEta = 999.;
    if ( jetEta.size() > 1 ) {
      for (unsigned int i = 0; i < jetEta.size(); i++){
        for (unsigned int j = i+1; j < jetEta.size(); j++){
          deltaEta = std::min(deltaEta,std::fabs(jetEta[i]-jetEta[j]));
        }
      }
    }

    genJetDeltaEtaMin->Fill(deltaEta,weight);
  }

  if(GenLeptons.size()>0 && GenNeutrinos.size()>0 ){
    std::sort(GenLeptons.begin(), GenLeptons.end(), HepMCValidationHelper::sortByPt); 
    std::sort(GenNeutrinos.begin(), GenNeutrinos.end(), HepMCValidationHelper::sortByPt);
    if(GenLeptons.size()>0)    leading_l_pt->Fill(GenLeptons[0]->momentum().perp(),weight);
    if(GenLeptons.size()>1)    subleading_l_pt->Fill(GenLeptons[1]->momentum().perp(),weight);
    if(GenLeptons.size()>2)    subsubleading_l_pt->Fill(GenLeptons[2]->momentum().perp(),weight);
    if(GenLeptons.size()>0)    leading_l_eta->Fill(GenLeptons[0]->momentum().eta(),weight);
    if(GenLeptons.size()>1)    subleading_l_eta->Fill(GenLeptons[1]->momentum().eta(),weight);
    if(GenLeptons.size()>2)    subsubleading_l_eta->Fill(GenLeptons[2]->momentum().eta(),weight);
  }
  if(GenLeptons.size()>1 ){
    for(unsigned int i = 0; i<GenLeptons.size();i++){
      for(unsigned int j = i; j<GenLeptons.size();j++){
        if(j==i)continue;
        TLorentzVector lep1(GenLeptons[i]->momentum().x(), GenLeptons[i]->momentum().y(), GenLeptons[i]->momentum().z(), GenLeptons[i]->momentum().t()); 
        TLorentzVector lep2(GenLeptons[j]->momentum().x(), GenLeptons[j]->momentum().y(), GenLeptons[j]->momentum().z(), GenLeptons[j]->momentum().t()); 
        mll->Fill((lep1+lep2).M(),weight);
        ptll->Fill((lep1+lep2).Pt(),weight);
      }
    }
  }
  if(GenLeptons.size()>2 && GenNeutrinos.size()>2 ){
    TLorentzVector lep1(GenLeptons[0]->momentum().x(), GenLeptons[0]->momentum().y(), GenLeptons[0]->momentum().z(), GenLeptons[0]->momentum().t()); 
    TLorentzVector lep2(GenLeptons[1]->momentum().x(), GenLeptons[1]->momentum().y(), GenLeptons[1]->momentum().z(), GenLeptons[1]->momentum().t()); 
    TLorentzVector lep3(GenLeptons[2]->momentum().x(), GenLeptons[2]->momentum().y(), GenLeptons[2]->momentum().z(), GenLeptons[2]->momentum().t());
    TLorentzVector nu1(GenNeutrinos[0]->momentum().x(), GenNeutrinos[0]->momentum().y(), GenNeutrinos[0]->momentum().z(), GenNeutrinos[0]->momentum().t()); 
    TLorentzVector nu2(GenNeutrinos[1]->momentum().x(), GenNeutrinos[1]->momentum().y(), GenNeutrinos[1]->momentum().z(), GenNeutrinos[1]->momentum().t()); 
    TLorentzVector nu3(GenNeutrinos[2]->momentum().x(), GenNeutrinos[2]->momentum().y(), GenNeutrinos[2]->momentum().z(), GenNeutrinos[2]->momentum().t()); 
    mlll->Fill((lep1+lep2+lep3).M(),weight);
    ptlll->Fill((lep1+lep2+lep3).Pt(),weight);
    mlllnununu->Fill((lep1+lep2+lep3+nu1+nu2+nu3).M(),weight);
    mtlllnununu->Fill((lep1+lep2+lep3+nu1+nu2+nu3).Mt(),weight);
    ptlllnununu->Fill((lep1+lep2+lep3+nu1+nu2+nu3).Pt(),weight);
		
  }
	
  delete myGenEvent;
}
int VVVValidation::getParentBarcode(HepMC::GenParticle* it)
{
    int id = 0;
    if ( (it)->production_vertex() && (it)-> status()==3) {
        for ( HepMC::GenVertex::particle_iterator mother 
                  = (it)->production_vertex()->particles_begin(HepMC::parents);mother != (it)->production_vertex()->particles_end(HepMC::parents); ++mother ) {

           if((fabs((*mother)->pdg_id())==24)) id = (*mother)->barcode();
        }
    }
    return id;
}
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE (VVVValidation);

