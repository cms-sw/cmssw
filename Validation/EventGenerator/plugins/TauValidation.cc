/*class TauValidation
 *  
 *  Class to fill dqm monitor elements from existing EDM file
 *
 *  $Date: 2012/11/03 12:05:15 $
 *  $Revision: 1.21 $
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
  _wmanager(iPSet)
  ,hepmcCollection_(iPSet.getParameter<edm::InputTag>("hepmcCollection"))
  ,tauEtCut(iPSet.getParameter<double>("tauEtCutForRtau"))
  ,NJAKID(22)
{    
  dbe = 0;
  dbe = edm::Service<DQMStore>().operator->();
}

TauValidation::~TauValidation() {}

void TauValidation::beginJob()
{
  if(dbe){
    ///Setting the DQM top directories
    dbe->setCurrentFolder("Generator/Tau");
    
    // Number of analyzed events
    nTaus = dbe->book1D("nTaus", "n analyzed Taus", 1, 0., 1.);
    nPrimeTaus = dbe->book1D("nPrimeTaus", "n analyzed prime Taus", 1, 0., 1.);

    //Kinematics
    TauPt            = dbe->book1D("TauPt","Tau pT", 100 ,0,100);
    TauEta           = dbe->book1D("TauEta","Tau eta", 100 ,-2.5,2.5);
    TauPhi           = dbe->book1D("TauPhi","Tau phi", 100 ,-3.14,3.14);
    TauProngs        = dbe->book1D("TauProngs","Tau n prongs", 7 ,0,7);
    TauDecayChannels = dbe->book1D("TauDecayChannels","Tau decay channels", 13 ,0,13);
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

    TauMothers        = dbe->book1D("TauMothers","Tau mother particles", 10 ,0,10);
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

    TauRtauW          = dbe->book1D("TauRtauW","W->Tau p(leading track)/E(visible tau)", 50 ,0,1);
	TauRtauW->setAxisTitle("rtau");
    TauRtauHpm        = dbe->book1D("TauRtauHpm","Hpm->Tau p(leading track)/E(visible tau)", 50 ,0,1);
	TauRtauHpm->setAxisTitle("rtau");
    TauSpinEffectsW   = dbe->book1D("TauSpinEffectsW","Pion energy in W rest frame", 50 ,0,1);
	TauSpinEffectsW->setAxisTitle("Energy");
    TauSpinEffectsHpm = dbe->book1D("TauSpinEffectsHpm","Pion energy in Hpm rest frame", 50 ,0,1);
	TauSpinEffectsHpm->setAxisTitle("Energy");
    TauSpinEffectsZ   = dbe->book1D("TauSpinEffectsZ","Mass of pi+ pi-", 22 ,0,1.1);
    TauSpinEffectsZ->setAxisTitle("M_{#pi^{+}#pi^{-}}");

    TauFSRPhotonsN=dbe->book1D("TauFSRPhotonsN","FSR Photons radiating from/with tau (Gauge Boson)", 5 ,-0.5,4.5);
    TauFSRPhotonsN->setAxisTitle("N FSR Photons radiating from/with tau");
    TauFSRPhotonsPt=dbe->book1D("TauFSRPhotonsPt","Pt of FSR Photons radiating from/with tau (Gauge Boson)", 100 ,0,100);
    TauFSRPhotonsPt->setAxisTitle("P_{t} of FSR Photons radiating from/with tau [per tau]");
    TauFSRPhotonsPtSum=dbe->book1D("TauFSRPhotonsPtSum","Pt of FSR Photons radiating from/with tau (Gauge Boson)", 100 ,0,100);
    TauFSRPhotonsPtSum->setAxisTitle("P_{t} of FSR Photons radiating from/with tau [per tau]");

    TauBremPhotonsN=dbe->book1D("TauBremPhotonsN","Brem. Photons radiating in tau decay", 5 ,-0.5,4.5);
    TauBremPhotonsN->setAxisTitle("N FSR Photons radiating from/with tau");
    TauBremPhotonsPt=dbe->book1D("TauBremPhotonsPt","Sum Brem Pt ", 100 ,0,100);
    TauFSRPhotonsPt->setAxisTitle("P_{t} of Brem. Photons radiating in tau decay");    
    TauBremPhotonsPtSum =dbe->book1D("TauBremPhotonsPtSum","Sum of Brem Pt ", 100 ,0,100);
    TauFSRPhotonsPtSum->setAxisTitle("Sum P_{t} of Brem. Photons radiating in tau decay");

    JAKID =dbe->book1D("JAKID","JAK ID",NJAKID+1,-0.5,NJAKID+0.5);
    for(unsigned int i=0; i<NJAKID+1;i++){
      JAKInvMass.push_back(std::vector<MonitorElement *>());
      TString tmp="JAKID";
      tmp+=i;
      JAKInvMass.at(i).push_back(dbe->book1D("M"+tmp,"M_{"+tmp+"} (GeV)", 80 ,0,2.0));
      if(i==TauDecay::JAK_A1_3PI ||
	 i==TauDecay::JAK_KPIK ||
	 i==TauDecay::JAK_KPIPI ){
	JAKInvMass.at(i).push_back(dbe->book1D("M13"+tmp,"M_{13,"+tmp+"} (GeV)", 80 ,0,2.0));
	JAKInvMass.at(i).push_back(dbe->book1D("M23"+tmp,"M_{23,"+tmp+"} (GeV)", 80 ,0,2.0));
	JAKInvMass.at(i).push_back(dbe->book1D("M12"+tmp,"M_{12,"+tmp+"} (GeV)", 80 ,0,2.0));
      }
    }
  }
  
  return;
}

void TauValidation::endJob(){
  return;
}

void TauValidation::beginRun(const edm::Run& iRun,const edm::EventSetup& iSetup)
{
  ///Get PDT Table
  iSetup.getData( fPDGTable );
  return;
}
void TauValidation::endRun(const edm::Run& iRun,const edm::EventSetup& iSetup){return;}
void TauValidation::analyze(const edm::Event& iEvent,const edm::EventSetup& iSetup)
{ 
  
  ///Gathering the HepMCProduct information
  edm::Handle<HepMCProduct> evt;
  iEvent.getByLabel(hepmcCollection_, evt);

  //Get EVENT
  HepMC::GenEvent *myGenEvent = new HepMC::GenEvent(*(evt->GetEvent()));

  double weight = _wmanager.weight(iEvent);

  // find taus
  for(HepMC::GenEvent::particle_const_iterator iter = myGenEvent->particles_begin(); iter != myGenEvent->particles_end(); iter++) {
    if(abs((*iter)->pdg_id())==23){
      spinEffectsZ(*iter,weight);
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
	  spinEffects(*iter,mother,decaychannel,weight);
	  photons(*iter,weight);
	}
	///////////////////////////////////////////////
	//Adding JAKID and Mass information
	//
	TauDecay_CMSSW TD;
	unsigned int jak_id, TauBitMask;
	TD.AnalyzeTau((*iter),jak_id,TauBitMask,false,false);
	JAKID->Fill(jak_id,weight);
	if(jak_id<=NJAKID){
	  int tcharge=(*iter)->pdg_id()/abs((*iter)->pdg_id());
	  std::vector<HepMC::GenParticle*> part=TD.Get_TauDecayProducts();
	  TLorentzVector LVQ(0,0,0,0);
	  TLorentzVector LVS12(0,0,0,0);
	  TLorentzVector LVS13(0,0,0,0);
	  TLorentzVector LVS23(0,0,0,0);
	  bool haspart1=false;
	  for(unsigned int i=0;i<part.size();i++){
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

void TauValidation::spinEffects(const HepMC::GenParticle* tau,int mother, int decay, double weight){

	if(decay != pi) return; // polarization only for 1-prong hadronic taus with no neutral pions

	TLorentzVector momP4 = motherP4(tau);
	TLorentzVector pionP4 = leadingPionP4(tau);

	pionP4.Boost(-1*momP4.BoostVector());

	double energy = pionP4.E()/(momP4.M()/2);

	if(abs(mother) == 24) TauSpinEffectsW->Fill(energy,weight);	
	if(abs(mother) == 37) TauSpinEffectsHpm->Fill(energy,weight);
}

void TauValidation::spinEffectsZ(const HepMC::GenParticle* boson, double weight){

  TLorentzVector tautau(0,0,0,0);
  TLorentzVector pipi(0,0,0,0);
  
  int nSinglePionDecays = 0;
  if ( boson->end_vertex() ) {
    HepMC::GenVertex::particle_iterator des;
    for(des = boson->end_vertex()->particles_begin(HepMC::descendants);
	des!= boson->end_vertex()->particles_end(HepMC::descendants);++des ) {
      
      int pid = (*des)->pdg_id();
      if(abs(findMother(*des)) != 15 &&
	 abs(pid) == 15 && tauDecayChannel(*des) == pi){
	nSinglePionDecays++;
	tautau += TLorentzVector((*des)->momentum().px(),
				 (*des)->momentum().py(),
				 (*des)->momentum().pz(),
				 (*des)->momentum().e());
	pipi += leadingPionP4(*des);
      }
    }
  }
  if(nSinglePionDecays == 2 && tautau.M() != 0) {
    TauSpinEffectsZ->Fill(pipi.M()/tautau.M(),weight);
  }
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

                        if(abs(pid) != 211) continue;
 
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

  TLorentzVector p4(0,0,0,0);
  
  if ( tau->production_vertex() ) {
    HepMC::GenVertex::particle_iterator mother;
    for (mother = tau->production_vertex()->particles_begin(HepMC::parents);
	 mother!= tau->production_vertex()->particles_end(HepMC::parents); ++mother ) {
      p4 = TLorentzVector((*mother)->momentum().px(),
			  (*mother)->momentum().py(),
			  (*mother)->momentum().pz(),
			  (*mother)->momentum().e());
    }
  }
  
  return p4;
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

