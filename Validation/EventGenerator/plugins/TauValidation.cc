/*class TauValidation
 *  
 *  Class to fill dqm monitor elements from existing EDM file
 *
 */
#include "Validation/EventGenerator/interface/TauValidation.h"

#include "CLHEP/Units/defs.h"
#include "CLHEP/Units/PhysicalConstants.h"

#include "DataFormats/Math/interface/LorentzVector.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "Validation/EventGenerator/interface/TauDecay_CMSSW.h"
#include "Validation/EventGenerator/interface/PdtPdgMini.h"

using namespace edm;

TauValidation::TauValidation(const edm::ParameterSet& iPSet): 
  wmanager_(iPSet,consumesCollector())
  ,hepmcCollection_(iPSet.getParameter<edm::InputTag>("hepmcCollection"))
  ,tauEtCut(iPSet.getParameter<double>("tauEtCutForRtau"))
  ,NJAKID(22)
  ,zsbins(20)
  ,zsmin(-0.5)
  ,zsmax(0.5)
{    
  hepmcCollectionToken_=consumes<HepMCProduct>(hepmcCollection_);
}

TauValidation::~TauValidation(){}

void TauValidation::dqmBeginRun(const edm::Run& r, const edm::EventSetup& c) {
  c.getData( fPDGTable );
}

void TauValidation::bookHistograms(DQMStore::IBooker &i, edm::Run const &, edm::EventSetup const &){
    ///Setting the DQM top directories
    i.setCurrentFolder("Generator/Tau");
    
    // Number of analyzed events
    nTaus = i.book1D("nTaus", "n analyzed Taus", 1, 0., 1.);
    nPrimeTaus = i.book1D("nPrimeTaus", "n analyzed prime Taus", 1, 0., 1.);

    //Kinematics
    TauPt            = i.book1D("TauPt","Tau pT", 100 ,0,100);
    TauEta           = i.book1D("TauEta","Tau eta", 100 ,-2.5,2.5);
    TauPhi           = i.book1D("TauPhi","Tau phi", 100 ,-3.14,3.14);
    TauProngs        = i.book1D("TauProngs","Tau n prongs", 7 ,0,7);
    TauDecayChannels = i.book1D("TauDecayChannels","Tau decay channels", 13 ,0,13);
	TauDecayChannels->setBinLabel(1+undetermined,"?");
    	TauDecayChannels->setBinLabel(1+electron,"e");
    	TauDecayChannels->setBinLabel(1+muon,"mu");
    	TauDecayChannels->setBinLabel(1+pi,"#pi^{#pm}");
	TauDecayChannels->setBinLabel(1+rho,"#rho^{#pm}");
	TauDecayChannels->setBinLabel(1+a1,"a_{1}^{#pm}");
	TauDecayChannels->setBinLabel(1+pi1pi0,"#pi^{#pm}#pi^{0}");
    	TauDecayChannels->setBinLabel(1+pinpi0,"#pi^{#pm}n#pi^{0}");
    	TauDecayChannels->setBinLabel(1+tripi,"3#pi^{#pm}");
    	TauDecayChannels->setBinLabel(1+tripinpi0,"3#pi^{#pm}n#pi^{0}");
	TauDecayChannels->setBinLabel(1+K,"K");
	TauDecayChannels->setBinLabel(1+Kstar,"K^{*}");
	TauDecayChannels->setBinLabel(1+stable,"Stable");

    TauMothers        = i.book1D("TauMothers","Tau mother particles", 10 ,0,10);
	TauMothers->setBinLabel(1+other,"?");
	TauMothers->setBinLabel(1+B,"B Decays");
	TauMothers->setBinLabel(1+D,"D Decays");
	TauMothers->setBinLabel(1+gamma,"#gamma");
	TauMothers->setBinLabel(1+Z,"Z");
	TauMothers->setBinLabel(1+W,"W");
	TauMothers->setBinLabel(1+HSM,"H_{SM}/h^{0}");
	TauMothers->setBinLabel(1+H0,"H^{0}");
	TauMothers->setBinLabel(1+A0,"A^{0}");
	TauMothers->setBinLabel(1+Hpm,"H^{#pm}");

	DecayLength = i.book1D("DecayLength","#tau Decay Length", 100 ,0,20);  DecayLength->setAxisTitle("L_{#tau} (mm)");
	LifeTime =  i.book1D("LifeTime","#tau LifeTime ", 100 ,0,1000E-15);     LifeTime->setAxisTitle("#tau_{#tau}s)");

    TauRtauW          = i.book1D("TauRtauW","W->Tau p(leading track)/E(visible tau)", 50 ,0,1);     TauRtauW->setAxisTitle("rtau");
    TauRtauHpm        = i.book1D("TauRtauHpm","Hpm->Tau p(leading track)/E(visible tau)", 50 ,0,1); TauRtauHpm->setAxisTitle("rtau");

    TauSpinEffectsW_X   = i.book1D("TauSpinEffectsWX","Pion energy in W rest frame", 50 ,0,1);     TauSpinEffectsW_X->setAxisTitle("X");
    TauSpinEffectsHpm_X = i.book1D("TauSpinEffectsHpmX","Pion energy in Hpm rest frame", 50 ,0,1); TauSpinEffectsHpm_X->setAxisTitle("X");

    TauSpinEffectsW_eX   = i.book1D("TauSpinEffectsWeX","e energy in W rest frame", 50 ,0,1);     TauSpinEffectsW_eX->setAxisTitle("X");
    TauSpinEffectsHpm_eX = i.book1D("TauSpinEffectsHpmeX","e energy in Hpm rest frame", 50 ,0,1); TauSpinEffectsHpm_eX->setAxisTitle("X");

    TauSpinEffectsW_muX   = i.book1D("TauSpinEffectsWmuX","mu energy in W rest frame", 50 ,0,1);     TauSpinEffectsW_muX->setAxisTitle("X");
    TauSpinEffectsHpm_muX = i.book1D("TauSpinEffectsHpmmuX","mu energy in Hpm rest frame", 50 ,0,1); TauSpinEffectsHpm_muX->setAxisTitle("X");

    TauSpinEffectsW_UpsilonRho   = i.book1D("TauSpinEffectsWUpsilonRho","#Upsilon for #rho", 50 ,-1,1);     TauSpinEffectsW_UpsilonRho->setAxisTitle("#Upsilon");
    TauSpinEffectsHpm_UpsilonRho = i.book1D("TauSpinEffectsHpmUpsilonRho","#Upsilon for #rho", 50 ,-1,1);   TauSpinEffectsHpm_UpsilonRho->setAxisTitle("#Upsilon");

    TauSpinEffectsW_UpsilonA1   = i.book1D("TauSpinEffectsWUpsilonA1","#Upsilon for a1", 50 ,-1,1);       TauSpinEffectsW_UpsilonA1->setAxisTitle("#Upsilon");
    TauSpinEffectsHpm_UpsilonA1 = i.book1D("TauSpinEffectsHpmUpsilonA1","#Upsilon for a1", 50 ,-1,1);     TauSpinEffectsHpm_UpsilonA1->setAxisTitle("#Upsilon");

    TauSpinEffectsH_pipiAcoplanarity = i.book1D("TauSpinEffectsH_pipiAcoplanarity","H Acoplanarity for #pi^{-}#pi^{+}", 50 ,0,2*TMath::Pi()); TauSpinEffectsH_pipiAcoplanarity->setAxisTitle("Acoplanarity"); 

    TauSpinEffectsH_pipiAcollinearity = i.book1D("TauSpinEffectsH_pipiAcollinearity","H Acollinearity for #pi^{-}#pi^{+}", 50 ,0,TMath::Pi()); TauSpinEffectsH_pipiAcollinearity->setAxisTitle("Acollinearity");
    TauSpinEffectsH_pipiAcollinearityzoom = i.book1D("TauSpinEffectsH_pipiAcollinearityzoom","H Acollinearity for #pi^{-}#pi^{+}", 50 ,3,TMath::Pi()); TauSpinEffectsH_pipiAcollinearityzoom->setAxisTitle("Acollinearity");

    TauSpinEffectsZ_MVis   = i.book1D("TauSpinEffectsZMVis","Mass of pi+ pi-", 25 ,0,1.1);       TauSpinEffectsZ_MVis->setAxisTitle("M_{#pi^{+}#pi^{-}}");
    TauSpinEffectsH_MVis   = i.book1D("TauSpinEffectsHMVis","Mass of pi+ pi-", 25 ,0,1.1);       TauSpinEffectsZ_MVis->setAxisTitle("M_{#pi^{+}#pi^{-}}");

    TauSpinEffectsZ_Zs   = i.book1D("TauSpinEffectsZZs","Z_{s}", zsbins ,zsmin,zsmax);        TauSpinEffectsZ_Zs->setAxisTitle("Z_{s}");
    TauSpinEffectsH_Zs   = i.book1D("TauSpinEffectsHZs","Z_{s}", zsbins ,zsmin,zsmax);        TauSpinEffectsZ_Zs->setAxisTitle("Z_{s}");

    TauSpinEffectsZ_X= i.book1D("TauSpinEffectsZX","X of #tau^{-}", 25 ,0,1.0);           TauSpinEffectsZ_X->setAxisTitle("X");
    TauSpinEffectsH_X= i.book1D("TauSpinEffectsH_X","X of #tau^{-}", 25 ,0,1.0);           TauSpinEffectsH_X->setAxisTitle("X");

    TauSpinEffectsZ_Xf   = i.book1D("TauSpinEffectsZXf","X of forward emitted #tau^{-}", 25 ,0,1.0);           TauSpinEffectsZ_Xf->setAxisTitle("X_{f}");
    TauSpinEffectsH_Xf   = i.book1D("TauSpinEffectsHXf","X of forward emitted #tau^{-}", 25 ,0,1.0);           TauSpinEffectsZ_Xf->setAxisTitle("X_{f}");

    TauSpinEffectsZ_Xb   = i.book1D("TauSpinEffectsZXb","X of backward emitted #tau^{-}", 25 ,0,1.0);           TauSpinEffectsZ_Xb->setAxisTitle("X_{b}");
    TauSpinEffectsH_Xb   = i.book1D("TauSpinEffectsHXb","X of backward emitted #tau^{-}", 25 ,0,1.0);           TauSpinEffectsZ_Xb->setAxisTitle("X_{b}");

    TauSpinEffectsZ_eX   = i.book1D("TauSpinEffectsZeX","e energy in Z rest frame", 50 ,0,1);     TauSpinEffectsZ_eX->setAxisTitle("X");
    TauSpinEffectsH_eX = i.book1D("TauSpinEffectsHeX","e energy in H rest frame", 50 ,0,1); TauSpinEffectsH_eX->setAxisTitle("X");

    TauSpinEffectsZ_muX   = i.book1D("TauSpinEffectsZmuX","mu energy in z rest frame", 50 ,0,1);     TauSpinEffectsZ_muX->setAxisTitle("X");
    TauSpinEffectsH_muX = i.book1D("TauSpinEffectsHmuX","mu energy in H rest frame", 50 ,0,1); TauSpinEffectsH_muX->setAxisTitle("X");

     TauSpinEffectsH_rhorhoAcoplanarityminus = i.book1D("TauSpinEffectsH_rhorhoAcoplanarityminus","#phi^{*-} (acoplanarity) for Higgs #rightarrow #rho-#rho (y_{1}*y_{2}<0)", 32 ,0,2*TMath::Pi());     TauSpinEffectsH_rhorhoAcoplanarityminus->setAxisTitle("#phi^{*-} (Acoplanarity)");
     TauSpinEffectsH_rhorhoAcoplanarityplus = i.book1D("TauSpinEffectsH_rhorhoAcoplanarityplus","#phi^{*+} (acoplanarity) for Higgs #rightarrow #rho-#rho (y_{1}*y_{2}>0)", 32 ,0,2*TMath::Pi());     TauSpinEffectsH_rhorhoAcoplanarityplus->setAxisTitle("#phi^{*+} (Acoplanarity)");

    TauFSRPhotonsN=i.book1D("TauFSRPhotonsN","FSR Photons radiating from/with tau (Gauge Boson)", 5 ,-0.5,4.5);
    TauFSRPhotonsN->setAxisTitle("N FSR Photons radiating from/with tau");
    TauFSRPhotonsPt=i.book1D("TauFSRPhotonsPt","Pt of FSR Photons radiating from/with tau (Gauge Boson)", 100 ,0,100);
    TauFSRPhotonsPt->setAxisTitle("P_{t} of FSR Photons radiating from/with tau [per tau]");
    TauFSRPhotonsPtSum=i.book1D("TauFSRPhotonsPtSum","Pt of FSR Photons radiating from/with tau (Gauge Boson)", 100 ,0,100);
    TauFSRPhotonsPtSum->setAxisTitle("P_{t} of FSR Photons radiating from/with tau [per tau]");

    TauBremPhotonsN=i.book1D("TauBremPhotonsN","Brem. Photons radiating in tau decay", 5 ,-0.5,4.5);
    TauBremPhotonsN->setAxisTitle("N FSR Photons radiating from/with tau");
    TauBremPhotonsPt=i.book1D("TauBremPhotonsPt","Sum Brem Pt ", 100 ,0,100);
    TauFSRPhotonsPt->setAxisTitle("P_{t} of Brem. Photons radiating in tau decay");    
    TauBremPhotonsPtSum =i.book1D("TauBremPhotonsPtSum","Sum of Brem Pt ", 100 ,0,100);
    TauFSRPhotonsPtSum->setAxisTitle("Sum P_{t} of Brem. Photons radiating in tau decay");

    JAKID =i.book1D("JAKID","JAK ID",NJAKID+1,-0.5,NJAKID+0.5);
    for(unsigned int j=0; j<NJAKID+1;j++){
      JAKInvMass.push_back(std::vector<MonitorElement *>());
      TString tmp="JAKID";
      tmp+=j;
      JAKInvMass.at(j).push_back(i.book1D("M"+tmp,"M_{"+tmp+"} (GeV)", 80 ,0,2.0));
      if(j==TauDecay::JAK_A1_3PI ||
	 j==TauDecay::JAK_KPIK ||
	 j==TauDecay::JAK_KPIPI ){
	JAKInvMass.at(j).push_back(i.book1D("M13"+tmp,"M_{13,"+tmp+"} (GeV)", 80 ,0,2.0));
	JAKInvMass.at(j).push_back(i.book1D("M23"+tmp,"M_{23,"+tmp+"} (GeV)", 80 ,0,2.0));
	JAKInvMass.at(j).push_back(i.book1D("M12"+tmp,"M_{12,"+tmp+"} (GeV)", 80 ,0,2.0));
      }
    }
  return;
}


void TauValidation::analyze(const edm::Event& iEvent,const edm::EventSetup& iSetup)
{ 
    ///Gathering the HepMCProduct information
  edm::Handle<HepMCProduct> evt;
  iEvent.getByToken(hepmcCollectionToken_, evt);

  //Get EVENT
  HepMC::GenEvent *myGenEvent = new HepMC::GenEvent(*(evt->GetEvent()));

  double weight =   wmanager_.weight(iEvent);

  //////////////////////////////////////////////
  /*
  edm::Handle<double> WT;
  iEvent.getByLabel(edm::InputTag("TauSpinnerGen","TauSpinnerWT"),WT);
  weight = 1.0;
  if(*(WT.product())>1e-3 && *(WT.product())<=10.0) weight=(*(WT.product()));
  else {weight=1.0;}
  */
  ///////////////////////////////////////////////

  // find taus
  for(HepMC::GenEvent::particle_const_iterator iter = myGenEvent->particles_begin(); iter != myGenEvent->particles_end(); iter++) {
     if(abs((*iter)->pdg_id())==PdtPdgMini::Z0 || abs((*iter)->pdg_id())==PdtPdgMini::Higgs0){
       spinEffectsZH(*iter,weight);
    }
    if(abs((*iter)->pdg_id())==15){
      if(isLastTauinChain(*iter)){
	nTaus->Fill(0.5,weight);
	int mother  = tauMother(*iter,weight);
	int decaychannel = tauDecayChannel(*iter,weight);
        tauProngs(*iter, weight);
	if(mother>-1){ // exclude B, D and other non-signal decay modes
	  nPrimeTaus->Fill(0.5,weight);
	  TauPt->Fill((*iter)->momentum().perp(),weight);
	  TauEta->Fill((*iter)->momentum().eta(),weight);
	  TauPhi->Fill((*iter)->momentum().phi(),weight);
	  rtau(*iter,mother,decaychannel,weight);
	  photons(*iter,weight);
	}
	///////////////////////////////////////////////
	//Adding JAKID and Mass information
	//
        TauDecay_CMSSW TD;
        unsigned int jak_id, TauBitMask;
        if(TD.AnalyzeTau((*iter),jak_id,TauBitMask,false,false)){
	  JAKID->Fill(jak_id,weight);
	  if(jak_id<=NJAKID){
	    int tcharge=(*iter)->pdg_id()/abs((*iter)->pdg_id());
	    std::vector<HepMC::GenParticle*> part=TD.Get_TauDecayProducts();
	     spinEffectsWHpm(*iter,mother,jak_id,part,weight);
	    TLorentzVector LVQ(0,0,0,0);
	    TLorentzVector LVS12(0,0,0,0);
	    TLorentzVector LVS13(0,0,0,0);
	    TLorentzVector LVS23(0,0,0,0);
	    bool haspart1=false;
	     TVector3 PV,SV;
	     bool hasDL(false);
	    for(unsigned int i=0;i<part.size();i++){
	       if(abs(part.at(i)->pdg_id())!=PdtPdgMini::nu_tau && TD.isTauFinalStateParticle(part.at(i)->pdg_id()) && !hasDL){
		 PV=TVector3((*iter)->production_vertex()->point3d().x(),(*iter)->production_vertex()->point3d().y(),(*iter)->production_vertex()->point3d().z());
		 SV=TVector3(part.at(i)->production_vertex()->point3d().x(),part.at(i)->production_vertex()->point3d().y(),part.at(i)->production_vertex()->point3d().z());
		 TVector3 DL=SV-PV;
		 DecayLength->Fill(DL.Mag(),weight);
		 double c(2.99792458E8),Ltau(DL.Mag()/1000),beta((*iter)->momentum().rho()/(*iter)->momentum().m());
		 LifeTime->Fill( Ltau/(c*beta),weight);
	       }

	      if(TD.isTauFinalStateParticle(part.at(i)->pdg_id()) &&
		 abs(part.at(i)->pdg_id())!=PdtPdgMini::nu_e &&
		 abs(part.at(i)->pdg_id())!=PdtPdgMini::nu_mu &&
		 abs(part.at(i)->pdg_id())!=PdtPdgMini::nu_tau ){
		TLorentzVector LV(part.at(i)->momentum().px(),part.at(i)->momentum().py(),part.at(i)->momentum().pz(),part.at(i)->momentum().e());
		LVQ+=LV;
		if(jak_id==TauDecay::JAK_A1_3PI ||
		   jak_id==TauDecay::JAK_KPIK ||
		   jak_id==TauDecay::JAK_KPIPI
		   ){
		  if((tcharge==part.at(i)->pdg_id()/abs(part.at(i)->pdg_id()) && TD.nProng(TauBitMask)==3) || (jak_id==TauDecay::JAK_A1_3PI && TD.nProng(TauBitMask)==1 && abs(part.at(i)->pdg_id())==PdtPdgMini::pi_plus) ){
		    LVS13+=LV;
		    LVS23+=LV;
		  }
		  else{
		    LVS12+=LV;
		    if(!haspart1 && ((jak_id==TauDecay::JAK_A1_3PI)  || (jak_id!=TauDecay::JAK_A1_3PI && abs(part.at(i)->pdg_id())==PdtPdgMini::K_plus) )){
		      LVS13+=LV;
		      haspart1=true;
		    }
		    else{
		      LVS23+=LV;
		    }
		  }
		}
	      }
	    }
	    part.clear();
	    JAKInvMass.at(jak_id).at(0)->Fill(LVQ.M(),weight);
	    if(jak_id==TauDecay::JAK_A1_3PI ||
	       jak_id==TauDecay::JAK_KPIK ||
	       jak_id==TauDecay::JAK_KPIPI
	       ){
	      JAKInvMass.at(jak_id).at(1)->Fill(LVS13.M(),weight);
	      JAKInvMass.at(jak_id).at(2)->Fill(LVS23.M(),weight);
	      JAKInvMass.at(jak_id).at(3)->Fill(LVS12.M(),weight);
	    }
	  }
	}
	else{
	  JAKID->Fill(jak_id,weight);  
	}
      }
    }
  }
  delete myGenEvent;
}//analyze

const HepMC::GenParticle* TauValidation::GetMother(const HepMC::GenParticle* tau){
  if ( tau->production_vertex() ) {
    HepMC::GenVertex::particle_iterator mother;
    for (mother = tau->production_vertex()->particles_begin(HepMC::parents); mother!= tau->production_vertex()->particles_end(HepMC::parents); mother++ ) {
      if((*mother)->pdg_id() == tau->pdg_id()) return GetMother(*mother);
      return (*mother);
    }
  }
  return tau;
}


const std::vector<HepMC::GenParticle*> TauValidation::GetMothers(const HepMC::GenParticle* boson){
  std::vector<HepMC::GenParticle*> mothers;
  if ( boson->production_vertex() ) {
    HepMC::GenVertex::particle_iterator mother;
    for (mother = boson->production_vertex()->particles_begin(HepMC::parents); mother!= boson->production_vertex()->particles_end(HepMC::parents); mother++ ) {
      if((*mother)->pdg_id() == boson->pdg_id()) return GetMothers(*mother);
      mothers.push_back(*mother);
    }
  }
  return mothers;
}


int TauValidation::findMother(const HepMC::GenParticle* tau){
  int mother_pid = 0;
  if ( tau->production_vertex() ) {
    HepMC::GenVertex::particle_iterator mother;
    for (mother = tau->production_vertex()->particles_begin(HepMC::parents); mother!= tau->production_vertex()->particles_end(HepMC::parents); mother++ ) {
      mother_pid = (*mother)->pdg_id();
      if(mother_pid == tau->pdg_id()) return findMother(*mother); //mother_pid = -1; Make recursive to look for last tau in chain
    }
  }
  return mother_pid;
}


bool TauValidation::isLastTauinChain(const HepMC::GenParticle* tau){
  if ( tau->end_vertex() ) {
    HepMC::GenVertex::particle_iterator dau;
    for (dau = tau->end_vertex()->particles_begin(HepMC::children); dau!= tau->end_vertex()->particles_end(HepMC::children); dau++ ) {
      int dau_pid = (*dau)->pdg_id();
      if(dau_pid == tau->pdg_id()) return false;
    }
  }
  return true;
}


void TauValidation::findTauList(const HepMC::GenParticle* tau,std::vector<const HepMC::GenParticle*> &TauList){
  TauList.insert(TauList.begin(),tau);
  if ( tau->production_vertex() ) {
    HepMC::GenVertex::particle_iterator mother;
    for (mother = tau->production_vertex()->particles_begin(HepMC::parents); mother!= tau->production_vertex()->particles_end(HepMC::parents);mother++) {
      if((*mother)->pdg_id() == tau->pdg_id()){
	findTauList(*mother,TauList);
      }
    }
  }
}

void TauValidation::findFSRandBrem(const HepMC::GenParticle* p, bool doBrem, std::vector<const HepMC::GenParticle*> &ListofFSR,
				  std::vector<const HepMC::GenParticle*> &ListofBrem){
  // note this code split the FSR and Brem based one if the tau decays into a tau+photon or not with the Fortran Tauola Interface, this is not 100% correct because photos puts the tau with the regular tau decay products. 
  if(abs(p->pdg_id())==15){
    if(isLastTauinChain(p)){ doBrem=true;}
    else{ doBrem=false;}
  }
    int photo_ID=22;
  if ( p->end_vertex() ) {
    HepMC::GenVertex::particle_iterator dau;
    for (dau = p->end_vertex()->particles_begin(HepMC::children); dau!= p->end_vertex()->particles_end(HepMC::children); dau++ ) {
      //if(doBrem) std::cout << "true " << (*dau)->pdg_id() << " "  << std::endl;
      //else std::cout << "false " << (*dau)->pdg_id() << " " << std::endl;
      if(abs((*dau)->pdg_id()) == abs(photo_ID) && !doBrem){ListofFSR.push_back(*dau);}
      if(abs((*dau)->pdg_id()) == abs(photo_ID) && doBrem){ListofBrem.push_back(*dau);}
      if((*dau)->end_vertex() && (*dau)->end_vertex()->particles_out_size()>0 && abs((*dau)->pdg_id()) != 111 && abs((*dau)->pdg_id()) != 221/* remove pi0 and eta decays*/){
	findFSRandBrem(*dau,doBrem,ListofFSR,ListofBrem);
      }
    }
  }
}



void TauValidation::FindPhotosFSR(const HepMC::GenParticle* p,std::vector<const HepMC::GenParticle*> &ListofFSR,double &BosonScale){
  BosonScale=0.0;
  const HepMC::GenParticle* m=GetMother(p);
  double mother_pid=m->pdg_id();
  if(m->end_vertex() && mother_pid!=p->pdg_id()){
    HepMC::GenVertex::particle_iterator dau;
    for (dau = m->end_vertex()->particles_begin(HepMC::children); dau!= m->end_vertex()->particles_end(HepMC::children); dau++ ) {
      int dau_pid = (*dau)->pdg_id();
      if(fabs(dau_pid) == 22) {
	ListofFSR.push_back(*dau);
      }
    }
  }
  if(abs(mother_pid) == 24) BosonScale=1.0; // W
  if(abs(mother_pid) == 23) BosonScale=2.0; // Z;
  if(abs(mother_pid) == 22) BosonScale=2.0; // gamma;
  if(abs(mother_pid) == 25) BosonScale=2.0; // HSM;
  if(abs(mother_pid) == 35) BosonScale=2.0; // H0;
  if(abs(mother_pid) == 36) BosonScale=2.0; // A0;
  if(abs(mother_pid) == 37) BosonScale=1.0; //Hpm;
}


int TauValidation::tauMother(const HepMC::GenParticle* tau, double weight){

  if(abs(tau->pdg_id()) != 15 ) return -3;
  
  int mother_pid = findMother(tau);
  if(mother_pid == -2) return -2;
  
  int label = other;
  if(abs(mother_pid) == 24) label = W;
  if(abs(mother_pid) == 23) label = Z;
  if(abs(mother_pid) == 22) label = gamma;
  if(abs(mother_pid) == 25) label = HSM;
  if(abs(mother_pid) == 35) label = H0;
  if(abs(mother_pid) == 36) label = A0;
  if(abs(mother_pid) == 37) label = Hpm;
  
  int mother_shortpid=(abs(mother_pid)%10000);
  if(mother_shortpid>500 && mother_shortpid<600 )label = B;
  if(mother_shortpid>400 && mother_shortpid<500)label = D;
  TauMothers->Fill(label,weight);
  if(label==B || label == D || label == other) return -1;
  
  return mother_pid;
}

int TauValidation::tauProngs(const HepMC::GenParticle* tau, double weight){
  int nProngs = 0;
  if ( tau->end_vertex() ) {
    HepMC::GenVertex::particle_iterator des;
    for(des = tau->end_vertex()->particles_begin(HepMC::descendants);
	des!= tau->end_vertex()->particles_end(HepMC::descendants);++des ) {
      int pid = (*des)->pdg_id();
      if(abs(pid) == 15) return tauProngs(*des, weight);
      if((*des)->status() != 1) continue; // dont count unstable particles
      
      const HepPDT::ParticleData*  pd = fPDGTable->particle((*des)->pdg_id ());
      int charge = (int) pd->charge();
      if(charge == 0) continue;
      nProngs++;
    }
  }
  TauProngs->Fill(nProngs,weight);
  return nProngs;
}

int TauValidation::tauDecayChannel(const HepMC::GenParticle* tau, double weight){

  int channel = undetermined;
  if(tau->status() == 1) channel = stable;
  int allCount   = 0,
    eCount     = 0,
    muCount    = 0,
    pi0Count   = 0,
    piCount    = 0,
    rhoCount   = 0,
    a1Count    = 0,
    KCount     = 0,
    KstarCount = 0;
  
  if ( tau->end_vertex() ) {
    HepMC::GenVertex::particle_iterator des;
    for(des = tau->end_vertex()->particles_begin(HepMC::descendants);
	des!= tau->end_vertex()->particles_end(HepMC::descendants);++des ) {
      int pid = (*des)->pdg_id();
      if(abs(pid) == 15) return tauDecayChannel(*des,weight);
      
      allCount++;
      if(abs(pid) == 11)    eCount++;
      if(abs(pid) == 13)    muCount++;
      if(abs(pid) == 111)   pi0Count++;
      if(abs(pid) == 211)   piCount++;
      if(abs(pid) == 213)   rhoCount++;
      if(abs(pid) == 20213) a1Count++;
      if(abs(pid) == 321)   KCount++;
      if(abs(pid) == 323)   KstarCount++;
      
    }
  }
  // resonances  
  if(KCount >= 1)     channel = K;
  if(KstarCount >= 1) channel = Kstar;
  if(a1Count >= 1)    channel = a1;
  if(rhoCount >= 1)   channel = rho;
  if(channel!=undetermined && weight!=0.0) TauDecayChannels->Fill(channel,weight);
  
  // final state products
  if(piCount == 1 && pi0Count == 0) channel = pi;
  if(piCount == 1 && pi0Count == 1) channel = pi1pi0;
  if(piCount == 1 && pi0Count > 1)  channel = pinpi0;
  if(piCount == 3 && pi0Count == 0) channel = tripi;
  if(piCount == 3 && pi0Count > 0)  channel = tripinpi0;
  if(eCount == 1)                   channel = electron;
  if(muCount == 1)                  channel = muon;
  if(weight!=0.0) TauDecayChannels->Fill(channel,weight);
  return channel;
}


void TauValidation::rtau(const HepMC::GenParticle* tau,int mother, int decay, double weight){

	if(decay != pi1pi0) return; // polarization only for 1-prong hadronic taus with one neutral pion to make a clean case

	if(tau->momentum().perp() < tauEtCut) return; // rtau visible only for boosted taus
	
	double rTau = 0;
	double ltrack = leadingPionMomentum(tau, weight);
	double visibleTauE = visibleTauEnergy(tau);

	if(visibleTauE != 0) rTau = ltrack/visibleTauE;

	if(abs(mother) == 24) TauRtauW->Fill(rTau,weight);
        if(abs(mother) == 37) TauRtauHpm->Fill(rTau,weight); 
}

 void TauValidation::spinEffectsWHpm(const HepMC::GenParticle* tau,int mother, int decay, std::vector<HepMC::GenParticle*> &part,double weight){
  if(decay == TauDecay::JAK_PION || decay == TauDecay::JAK_MUON || decay == TauDecay::JAK_ELECTRON){  // polarization only for 1-prong hadronic taus with no neutral pions
    TLorentzVector momP4 = motherP4(tau);
    TLorentzVector pionP4 = leadingPionP4(tau);
    pionP4.Boost(-1*momP4.BoostVector());
    double energy = pionP4.E()/(momP4.M()/2);
    if(decay == TauDecay::JAK_PION){
      if(abs(mother) == 24) TauSpinEffectsW_X->Fill(energy,weight);	
      if(abs(mother) == 37) TauSpinEffectsHpm_X->Fill(energy,weight);
    }
    if(decay == TauDecay::JAK_MUON){
      if(abs(mother) == 24) TauSpinEffectsW_muX->Fill(energy,weight);
      if(abs(mother) == 37) TauSpinEffectsHpm_muX->Fill(energy,weight);
    }
    if(decay == TauDecay::JAK_ELECTRON){
      if(abs(mother) == 24) TauSpinEffectsW_eX->Fill(energy,weight);
      if(abs(mother) == 37) TauSpinEffectsHpm_eX->Fill(energy,weight);
    }

  }
  else if(decay==TauDecay::JAK_RHO_PIPI0){
    TLorentzVector rho(0,0,0,0),pi(0,0,0,0);
    for(unsigned int i=0;i<part.size();i++){
      TLorentzVector LV(part.at(i)->momentum().px(),part.at(i)->momentum().py(),part.at(i)->momentum().pz(),part.at(i)->momentum().e());
      if(abs(part.at(i)->pdg_id())==PdtPdgMini::pi_plus){pi+=LV; rho+=LV;}
      if(abs(part.at(i)->pdg_id())==PdtPdgMini::pi0){rho+=LV;}
    }
    if(abs(mother) == 24) TauSpinEffectsW_UpsilonRho->Fill(2*pi.Pt()/rho.Pt()-1,weight);
    if(abs(mother) == 37) TauSpinEffectsHpm_UpsilonRho->Fill(2*pi.Pt()/rho.Pt()-1,weight);
  }
  else if(decay==TauDecay::JAK_A1_3PI){ // only for pi2pi0 for now
    TLorentzVector a1(0,0,0,0),pi_p(0,0,0,0),pi_m(0,0,0,0); 
    int nplus(0),nminus(0);
    for(unsigned int i=0;i<part.size();i++){
      TLorentzVector LV(part.at(i)->momentum().px(),part.at(i)->momentum().py(),part.at(i)->momentum().pz(),part.at(i)->momentum().e());
      if(part.at(i)->pdg_id()==PdtPdgMini::pi_plus){ pi_p+=LV; a1+=LV; nplus++;}
      if(part.at(i)->pdg_id()==PdtPdgMini::pi_minus){pi_m+=LV; a1+=LV; nminus++;}
    }
    double gamma=0;
    if(nplus+nminus==3 && nplus==1)  gamma=2*pi_p.Pt()/a1.Pt()-1;
    if(nplus+nminus==3 && nminus==1) gamma=2*pi_m.Pt()/a1.Pt()-1;
    else{
      pi_p+=pi_m; gamma=2*pi_p.Pt()/a1.Pt()-1;
    }
    if(abs(mother) == 24) TauSpinEffectsW_UpsilonA1->Fill(gamma,weight);
    if(abs(mother) == 37) TauSpinEffectsHpm_UpsilonA1->Fill(gamma,weight);
  }
}

 void TauValidation::spinEffectsZH(const HepMC::GenParticle* boson, double weight){
  TLorentzVector tautau(0,0,0,0);
  TLorentzVector pipi(0,0,0,0);
  TLorentzVector taum(0,0,0,0);
   TLorentzVector taup(0,0,0,0);
   TLorentzVector rho_plus,rho_minus,pi_rhominus,pi0_rhominus,pi_rhoplus,pi0_rhoplus,pi_plus,pi_minus;
   bool hasrho_minus(false),hasrho_plus(false),haspi_minus(false),haspi_plus(false);
  int nSinglePionDecays(0),nSingleMuonDecays(0),nSingleElectronDecays(0);
  double x1(0),x2(0); 
  TLorentzVector Zboson(boson->momentum().px(),boson->momentum().py(),boson->momentum().pz(),boson->momentum().e());
  if ( boson->end_vertex() ) {
    HepMC::GenVertex::particle_iterator des;
    for(des = boson->end_vertex()->particles_begin(HepMC::children);	des!= boson->end_vertex()->particles_end(HepMC::children);++des ) {
      int pid = (*des)->pdg_id();
      if(abs(findMother(*des)) != 15 &&	 abs(pid) == 15 && (tauDecayChannel(*des) == pi || tauDecayChannel(*des) == muon || tauDecayChannel(*des) == electron )){
	if(tauDecayChannel(*des) == pi)nSinglePionDecays++;
	if(tauDecayChannel(*des) == muon)nSingleMuonDecays++;
	if(tauDecayChannel(*des) == electron)nSingleElectronDecays++;
	TLorentzVector LVtau((*des)->momentum().px(),(*des)->momentum().py(),(*des)->momentum().pz(),(*des)->momentum().e());
	tautau += LVtau;
	TLorentzVector LVpi=leadingPionP4(*des);
	pipi+=LVpi;
	const HepPDT::ParticleData*  pd = fPDGTable->particle((*des)->pdg_id ());
	int charge = (int) pd->charge();
	LVtau.Boost(-1*Zboson.BoostVector());
	LVpi.Boost(-1*Zboson.BoostVector());
	if(tauDecayChannel(*des) == pi){
	  if(abs(boson->pdg_id())==PdtPdgMini::Z0) TauSpinEffectsZ_X->Fill(LVpi.P()/LVtau.E(),weight);
	  if(abs(boson->pdg_id())==PdtPdgMini::Higgs0) TauSpinEffectsH_X->Fill(LVpi.P()/LVtau.E(),weight);
	}
	if(charge<0){x1=LVpi.P()/LVtau.E(); taum=LVtau;}
	else{ x2=LVpi.P()/LVtau.E();}
     }
       if(abs(findMother(*des)) != 15 &&  abs(pid) == 15 && (tauDecayChannel(*des) == rho || tauDecayChannel(*des) == pi1pi0) ){
	 if ( (*des)->end_vertex() ) {
	   HepMC::GenVertex::particle_iterator tauprod;
	   TLorentzVector LVtau((*des)->momentum().px(),(*des)->momentum().py(),(*des)->momentum().pz(),(*des)->momentum().e());
	   if(pid == 15)taum=LVtau; 
	   if(pid ==-15)taup=LVtau;
	   for(tauprod = (*des)->end_vertex()->particles_begin(HepMC::descendants); tauprod!= (*des)->end_vertex()->particles_end(HepMC::descendants);++tauprod ) {
	     int pid_d = (*tauprod)->pdg_id();
	     if(abs(pid_d)==211 || abs(pid_d)==111){
	       TLorentzVector LV((*tauprod)->momentum().px(),(*tauprod)->momentum().py(),(*tauprod)->momentum().pz(),(*tauprod)->momentum().e());
	       if(pid==15){
		 hasrho_minus=true;
		 if(pid_d==-211 ){ pi_rhominus=LV;}
		 if(abs(pid_d)==111 ){ pi0_rhominus=LV;}
	       }
	       if(pid==-15){
		 hasrho_plus=true;
		 if(pid_d==211 ){pi_rhoplus=LV;}
		 if(abs(pid_d)==111 ){pi0_rhoplus=LV;} 
	       }
	     }
	   }
	 }
       }
       if(abs(findMother(*des)) != 15 &&  abs(pid) == 15 && (tauDecayChannel(*des) == pi) ){
         if ( (*des)->end_vertex() ) {
	   HepMC::GenVertex::particle_iterator tauprod;
           TLorentzVector LVtau((*des)->momentum().px(),(*des)->momentum().py(),(*des)->momentum().pz(),(*des)->momentum().e());
           if(pid == 15)taum=LVtau;
           if(pid ==-15)taup=LVtau;
           for(tauprod = (*des)->end_vertex()->particles_begin(HepMC::descendants); tauprod!= (*des)->end_vertex()->particles_end(HepMC::descendants);++tauprod ) {
             int pid_d = (*tauprod)->pdg_id();
             if(abs(pid_d)==211 || abs(pid_d)==111){
               TLorentzVector LV((*tauprod)->momentum().px(),(*tauprod)->momentum().py(),(*tauprod)->momentum().pz(),(*tauprod)->momentum().e());
               if(pid==15){
                 haspi_minus=true;
                 if(pid_d==-211 ){ pi_minus=LV;}
               }
               if(pid==-15){
                 haspi_plus=true;
                 if(pid_d==211 ){pi_plus=LV;}
               }
             }
           }
         }
    }
  }
   }
   if(hasrho_minus && hasrho_plus){
     //compute rhorho
     rho_minus=pi_rhominus;
     rho_minus+=pi0_rhominus;
     rho_plus=pi_rhoplus;
     rho_plus+=pi0_rhoplus;
     TLorentzVector rhorho=rho_minus;rhorho+=rho_plus;

     // boost to rhorho cm
     TLorentzVector pi_rhoplusb=pi_rhoplus;     pi_rhoplusb.Boost(-1*rhorho.BoostVector());
     TLorentzVector pi0_rhoplusb=pi0_rhoplus;   pi0_rhoplusb.Boost(-1*rhorho.BoostVector());
     TLorentzVector pi_rhominusb=pi_rhominus;   pi_rhominusb.Boost(-1*rhorho.BoostVector());
     TLorentzVector pi0_rhominusb=pi0_rhominus; pi0_rhominusb.Boost(-1*rhorho.BoostVector());
     
     // compute n+/-
     TVector3 n_plus=pi_rhoplusb.Vect().Cross(pi0_rhoplusb.Vect());
     TVector3 n_minus=pi_rhominusb.Vect().Cross(pi0_rhominusb.Vect());

     // compute the acoplanarity
     double Acoplanarity=acos(n_plus.Dot(n_minus)/(n_plus.Mag()*n_minus.Mag()));
     if(pi_rhominus.Vect().Dot(n_plus)>0){Acoplanarity*=-1;Acoplanarity+=2*TMath::Pi();}

     // now boost to tau frame
     pi_rhoplus.Boost(-1*taup.BoostVector());
     pi0_rhoplus.Boost(-1*taup.BoostVector());
     pi_rhominus.Boost(-1*taum.BoostVector());
     pi0_rhominus.Boost(-1*taum.BoostVector());

     // compute y1 and y2
     double y1=(pi_rhoplus.E()-pi0_rhoplus.E())/(pi_rhoplus.E()+pi0_rhoplus.E());
     double y2=(pi_rhominus.E()-pi0_rhominus.E())/(pi_rhominus.E()+pi0_rhominus.E());

     // fill histograms
     if(abs(boson->pdg_id())==PdtPdgMini::Higgs0 && y1*y2<0) TauSpinEffectsH_rhorhoAcoplanarityminus->Fill(Acoplanarity,weight);
     if(abs(boson->pdg_id())==PdtPdgMini::Higgs0 && y1*y2>0) TauSpinEffectsH_rhorhoAcoplanarityplus->Fill(Acoplanarity,weight);
   }
   if(haspi_minus && haspi_plus){
     TLorentzVector tauporig=taup;
     TLorentzVector taumorig=taum;

     // now boost to Higgs frame
     pi_plus.Boost(-1*Zboson.BoostVector());
     pi_minus.Boost(-1*Zboson.BoostVector());
     
     taup.Boost(-1*Zboson.BoostVector());
     taum.Boost(-1*Zboson.BoostVector());

     if(abs(boson->pdg_id())==PdtPdgMini::Higgs0){
       TauSpinEffectsH_pipiAcollinearity->Fill(acos(pi_plus.Vect().Dot(pi_minus.Vect())/(pi_plus.P()*pi_minus.P())));
       TauSpinEffectsH_pipiAcollinearityzoom->Fill(acos(pi_plus.Vect().Dot(pi_minus.Vect())/(pi_plus.P()*pi_minus.P())));
     }

     double proj_m=taum.Vect().Dot(pi_minus.Vect())/(taum.P()*taup.P());
     double proj_p=taup.Vect().Dot(pi_plus.Vect())/(taup.P()*taup.P());
     TVector3 Tau_m=taum.Vect();
     TVector3 Tau_p=taup.Vect();
     Tau_m*=proj_m;
     Tau_p*=proj_p;
     TVector3 Pit_m=pi_minus.Vect()-Tau_m;
     TVector3 Pit_p=pi_plus.Vect()-Tau_p;

     double Acoplanarity=acos(Pit_m.Dot(Pit_p)/(Pit_p.Mag()*Pit_m.Mag()));
     TVector3 n=Pit_p.Cross(Pit_m);
     if(n.Dot(Tau_m)/Tau_m.Mag()>0){Acoplanarity*=-1; Acoplanarity+=2*TMath::Pi();}
     // fill histograms
     if(abs(boson->pdg_id())==PdtPdgMini::Higgs0) TauSpinEffectsH_pipiAcoplanarity->Fill(Acoplanarity,weight);
     taup=tauporig;
     taum=taumorig;

   }


  if(nSingleMuonDecays==2){
    if(abs(boson->pdg_id())==PdtPdgMini::Z0)     TauSpinEffectsZ_muX->Fill(x1,weight);
    if(abs(boson->pdg_id())==PdtPdgMini::Higgs0) TauSpinEffectsH_muX->Fill(x1,weight);
  }
  if(nSingleElectronDecays==2){
    if(abs(boson->pdg_id())==PdtPdgMini::Z0)     TauSpinEffectsZ_eX->Fill(x1,weight);
    if(abs(boson->pdg_id())==PdtPdgMini::Higgs0) TauSpinEffectsH_eX->Fill(x1,weight);
  }
  if(nSinglePionDecays == 2 && tautau.M()!= 0) {
    for(int i=0;i<zsbins;i++){
      double zslow=((double)i)*(zsmax-zsmin)/((double)zsbins)+zsmin; 
      double zsup=((double)i+1)*(zsmax-zsmin)/((double)zsbins)+zsmin;
      double aup=Zstoa(zsup), alow=Zstoa(zslow);
      if(x2-x1>alow && x2-x1<aup){
	double zs=(zsup+zslow)/2;
	if(abs(boson->pdg_id())==PdtPdgMini::Z0)     TauSpinEffectsZ_Zs->Fill(zs,weight);
	if(abs(boson->pdg_id())==PdtPdgMini::Higgs0) TauSpinEffectsH_Zs->Fill(zs,weight);
	break;
      }
    }
    if(abs(boson->pdg_id())==PdtPdgMini::Z0)     TauSpinEffectsZ_MVis->Fill(pipi.M()/tautau.M(),weight);
    if(abs(boson->pdg_id())==PdtPdgMini::Higgs0) TauSpinEffectsH_MVis->Fill(pipi.M()/tautau.M(),weight);
 
    if(x1!=0){
      const std::vector<HepMC::GenParticle*> m=GetMothers(boson);
      int q(0),qbar(0);
      TLorentzVector Z(0,0,0,0);
      for(unsigned int i=0;i<m.size();i++){
	if(m.at(i)->pdg_id()==PdtPdgMini::d      || m.at(i)->pdg_id()==PdtPdgMini::u      ){q++;}
	if(m.at(i)->pdg_id()==PdtPdgMini::anti_d || m.at(i)->pdg_id()==PdtPdgMini::anti_u ){qbar++;}
      }
      if(q==1 && qbar==1){// assume q has largest E (valence vs see quarks) 
	if(taum.Vect().Dot(Zboson.Vect())/(Zboson.P()*taum.P())>0){
	  if(abs(boson->pdg_id())==PdtPdgMini::Z0)      TauSpinEffectsZ_Xf->Fill(x1,weight);
	  if(abs(boson->pdg_id())==PdtPdgMini::Higgs0) TauSpinEffectsH_Xf->Fill(x1,weight);
	}
	else{
	  if(abs(boson->pdg_id())==PdtPdgMini::Z0)      TauSpinEffectsZ_Xb->Fill(x1,weight);
	  if(abs(boson->pdg_id())==PdtPdgMini::Higgs0) TauSpinEffectsH_Xb->Fill(x1,weight);
	}
      }
    }
  }
}

double TauValidation::Zstoa(double zs){
  double a=1-sqrt(fabs(1.0-2*fabs(zs)));
  if(zs<0){
    a*=-1.0;
  }
  return a;
}


double TauValidation::leadingPionMomentum(const HepMC::GenParticle* tau, double weight){
	return leadingPionP4(tau).P();
}

TLorentzVector TauValidation::leadingPionP4(const HepMC::GenParticle* tau){

	TLorentzVector p4(0,0,0,0);

        if ( tau->end_vertex() ) {
              HepMC::GenVertex::particle_iterator des;
              for(des = tau->end_vertex()->particles_begin(HepMC::descendants);
                  des!= tau->end_vertex()->particles_end(HepMC::descendants);++des ) {
                        int pid = (*des)->pdg_id();

                        if(abs(pid) == 15) return leadingPionP4(*des);

                        if(!(abs(pid)==211 || abs(pid)==13 || abs(pid)==11)) continue;
 
			if((*des)->momentum().rho() > p4.P()) {
				p4 = TLorentzVector((*des)->momentum().px(),
                                                    (*des)->momentum().py(),
                                                    (*des)->momentum().pz(),
                                                    (*des)->momentum().e());
			} 
                }
        }

	return p4;
}

TLorentzVector TauValidation::motherP4(const HepMC::GenParticle* tau){
  const HepMC::GenParticle* m=GetMother(tau);
  return TLorentzVector(m->momentum().px(),m->momentum().py(),m->momentum().pz(),m->momentum().e());
}

double TauValidation::visibleTauEnergy(const HepMC::GenParticle* tau){
	TLorentzVector p4(tau->momentum().px(),
                          tau->momentum().py(),
                          tau->momentum().pz(),
                          tau->momentum().e());

        if ( tau->end_vertex() ) {
              HepMC::GenVertex::particle_iterator des;
              for(des = tau->end_vertex()->particles_begin(HepMC::descendants);
                  des!= tau->end_vertex()->particles_end(HepMC::descendants);++des ) {
                        int pid = (*des)->pdg_id();

                        if(abs(pid) == 15) return visibleTauEnergy(*des);

                        if(abs(pid) == 12 || abs(pid) == 14 || abs(pid) == 16) {
				p4 -= TLorentzVector((*des)->momentum().px(),
			                             (*des)->momentum().py(),
			                             (*des)->momentum().pz(),
			                             (*des)->momentum().e());
			}
                }
        }

	return p4.E();
}

void TauValidation::photons(const HepMC::GenParticle* tau, double weight){
  // Find First tau in chain
  std::vector<const HepMC::GenParticle*> TauList;
  findTauList(tau,TauList);

  // Get List of Gauge Boson to tau(s) FSR and Brem
  bool passedW=false;
  std::vector<const HepMC::GenParticle*> ListofFSR;  ListofFSR.clear();
  std::vector<const HepMC::GenParticle*> ListofBrem; ListofBrem.clear();
  std::vector<const HepMC::GenParticle*> FSR_photos; FSR_photos.clear();
  double BosonScale(1);
  if(TauList.size()>0){
    TauValidation::findFSRandBrem(TauList.at(0),passedW,ListofFSR,ListofBrem);
    TauValidation::FindPhotosFSR(TauList.at(0),FSR_photos,BosonScale);

    // Add the Tau Brem. information
    TauBremPhotonsN->Fill(ListofBrem.size(),weight);
    double photonPtSum=0;
    for(unsigned int i=0;i<ListofBrem.size();i++){
      photonPtSum+=ListofBrem.at(i)->momentum().perp();
      TauBremPhotonsPt->Fill(ListofBrem.at(i)->momentum().perp(),weight);
    }
    TauBremPhotonsPtSum->Fill(photonPtSum,weight);
        
    // Now add the Gauge Boson FSR information
    if(BosonScale!=0){
      TauFSRPhotonsN->Fill(ListofFSR.size(),weight);
      photonPtSum=0;
      for(unsigned int i=0;i<ListofFSR.size();i++){
	photonPtSum+=ListofFSR.at(i)->momentum().perp();
	TauFSRPhotonsPt->Fill(ListofFSR.at(i)->momentum().perp(),weight);
      }
      double FSR_photosSum(0);
      for(unsigned int i=0;i<FSR_photos.size();i++){
	FSR_photosSum+=FSR_photos.at(i)->momentum().perp();
	TauFSRPhotonsPt->Fill(FSR_photos.at(i)->momentum().perp()/BosonScale,weight*BosonScale);
      }
      TauFSRPhotonsPtSum->Fill(photonPtSum+FSR_photosSum/BosonScale,weight);
    }
  }
}

