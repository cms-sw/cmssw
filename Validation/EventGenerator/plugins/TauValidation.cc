/*class TauValidation
 *  
 *  Class to fill dqm monitor elements from existing EDM file
 *
 *  $Date: 2011/02/14 15:17:23 $
 *  $Revision: 1.11 $
 */
 
#include "Validation/EventGenerator/interface/TauValidation.h"

#include "CLHEP/Units/defs.h"
#include "CLHEP/Units/PhysicalConstants.h"

#include "DataFormats/Math/interface/LorentzVector.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

using namespace edm;

TauValidation::TauValidation(const edm::ParameterSet& iPSet):  
  hepmcCollection_(iPSet.getParameter<edm::InputTag>("hepmcCollection")),
  tauEtCut(iPSet.getParameter<double>("tauEtCutForRtau"))
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
    nEvt = dbe->book1D("nEvt", "n analyzed Events", 1, 0., 1.);

    //Kinematics
    TauPt            = dbe->book1D("TauPt","Tau pT", 100 ,0,100);
    TauEta           = dbe->book1D("TauEta","Tau eta", 100 ,-2.5,2.5);
    TauPhi           = dbe->book1D("TauPhi","Tau phi", 100 ,-3.14,3.14);
    TauProngs        = dbe->book1D("TauProngs","Tau n prongs", 7 ,0,7);
    TauDecayChannels = dbe->book1D("TauDecayChannels","Tau decay channels", 12 ,0,12);
	TauDecayChannels->setBinLabel(1+undetermined,"?");
    	TauDecayChannels->setBinLabel(1+electron,"e");
    	TauDecayChannels->setBinLabel(1+muon,"mu");
    	TauDecayChannels->setBinLabel(1+pi,"#pi^{#pm}");
	TauDecayChannels->setBinLabel(1+rho,"#rho^{#pm}");
	TauDecayChannels->setBinLabel(1+a1,"a_{1}^{#pm}");
	TauDecayChannels->setBinLabel(1+pi1pi0,"#pi^{#pm}#pi^{0}");
    	TauDecayChannels->setBinLabel(1+pinpi0,"#pi^{#pm}n#pi^{0}");
    	TauDecayChannels->setBinLabel(1+tripi,"3#pi^{#pm}");
//    	TauDecayChannels->setBinLabel(1+tripinpi0,"3#pi^{#pm}n#pi^{0}");
	TauDecayChannels->setBinLabel(1+K,"K");
	TauDecayChannels->setBinLabel(1+Kstar,"K^{*}");
	TauDecayChannels->setBinLabel(1+stable,"Stable");

    TauMothers        = dbe->book1D("TauMothers","Tau mother particles", 10 ,0,10);
	TauMothers->setBinLabel(1+other,"?");
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

    TauPhotonsN        = dbe->book1D("TauPhotonsN","Photons radiating from tau", 2 ,0,2);
	TauPhotonsN->setBinLabel(1,"Number of taus");
	TauPhotonsN->setBinLabel(2,"Number of taus radiating photons");
    TauPhotonsPt       = dbe->book1D("TauPhotonsPt","Photon pt radiating from tau", 2 ,0,2);
	TauPhotonsPt->setBinLabel(1,"Sum of tau pt");
	TauPhotonsPt->setBinLabel(2,"Sum of tau pt radiated by photons");
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

  nEvt->Fill(0.5);

  // find taus
  for(HepMC::GenEvent::particle_const_iterator iter = myGenEvent->particles_begin(); iter != myGenEvent->particles_end(); ++iter) {
      if(abs((*iter)->pdg_id())==15){
        TauPt->Fill((*iter)->momentum().perp());
        TauEta->Fill((*iter)->momentum().eta());
	TauPhi->Fill((*iter)->momentum().phi());
	int mother  = tauMother(*iter);
	int decaychannel = tauDecayChannel(*iter);
	tauProngs(*iter);
	rtau(*iter,mother,decaychannel);
	spinEffects(*iter,mother,decaychannel);
	photons(*iter);
      }
      if(abs((*iter)->pdg_id())==23){
        spinEffectsZ(*iter);
      }
  }

  delete myGenEvent;
}//analyze

int TauValidation::findMother(const HepMC::GenParticle* tau){
        int mother_pid = 0;

        if ( tau->production_vertex() ) {
                HepMC::GenVertex::particle_iterator mother;
                for (mother = tau->production_vertex()->particles_begin(HepMC::parents);
                     mother!= tau->production_vertex()->particles_end(HepMC::parents); ++mother ) {
                        mother_pid = (*mother)->pdg_id();
			if(mother_pid == tau->pdg_id()) mother_pid = -1;//findMother(*mother);
                        //std::cout << " parent " << mother_pid << std::endl;
                }
        }
	return mother_pid;
}

int TauValidation::tauMother(const HepMC::GenParticle* tau){

	if(abs(tau->pdg_id()) != 15 ) return -1;

	int mother_pid = findMother(tau);
	if(mother_pid == -1) return -1;

	int label = other;
	if(abs(mother_pid) == 24) label = W;
        if(abs(mother_pid) == 23) label = Z;
	if(abs(mother_pid) == 22) label = gamma;
	if(abs(mother_pid) == 25) label = HSM;
	if(abs(mother_pid) == 35) label = H0;
	if(abs(mother_pid) == 36) label = A0;
	if(abs(mother_pid) == 37) label = Hpm;

	TauMothers->Fill(label);

	return mother_pid;
}

int TauValidation::tauProngs(const HepMC::GenParticle* tau){

	int nProngs = 0;
	if ( tau->end_vertex() ) {
		HepMC::GenVertex::particle_iterator des;
		for(des = tau->end_vertex()->particles_begin(HepMC::descendants);
		    des!= tau->end_vertex()->particles_end(HepMC::descendants);++des ) {
			int pid = (*des)->pdg_id();
			if(abs(pid) == 15) return tauProngs(*des);
			if((*des)->status() != 1) continue; // dont count unstable particles

			const HepPDT::ParticleData*  pd = fPDGTable->particle((*des)->pdg_id ());
			int charge = (int) pd->charge();
			if(charge == 0) continue;
			//std::cout << "TauValidation::tauProngs barcode=" << (*des)->barcode() << " pid=" 
                        //          << pid << " mom=" << tauMother(*des) << " status=" 
                        //          << (*des)->status() << " charge=" << charge << std::endl;
			nProngs++;
		}
	}
	TauProngs->Fill(nProngs);
	return nProngs;
}

int TauValidation::findTauDecayChannel(const HepMC::GenParticle* tau){

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

                        if(abs(tauMother(*des)) != 15) continue;
                        int pid = (*des)->pdg_id();
                        //std::cout << " barcode=" << (*des)->barcode() << " pid="
                        //          << pid << " mom=" << tauMother(*des) << " status="
                        //          << (*des)->status() << std::endl;

                        if(abs(pid) == 15) return findTauDecayChannel(*des);

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

	if(KCount == 1 && allCount == 2)  channel = K;
	if(KstarCount == 1 && allCount == 2)  channel = Kstar;
	if(a1Count == 1 && allCount == 2)  channel = a1;
	if(rhoCount == 1 && allCount == 2)  channel = rho;

	if(piCount == 1 && pi0Count == 0) channel = pi;
	if(piCount == 1 && pi0Count == 1) channel = pi1pi0;
        if(piCount == 1 && pi0Count > 1)  channel = pinpi0;

        if(piCount == 3 && pi0Count == 0) channel = tripi;
//        if(piCount == 3 && pi0Count > 0)  channel = tripinpi0;

	if(eCount == 1)                   channel = electron;
	if(muCount == 1)                  channel = muon;

	return channel;
}

int TauValidation::tauDecayChannel(const HepMC::GenParticle* tau){
	int channel = findTauDecayChannel(tau);
	TauDecayChannels->Fill(channel);
	return channel;
}

void TauValidation::rtau(const HepMC::GenParticle* tau,int mother, int decay){

	if(decay != pi1pi0) return; // polarization only for 1-prong hadronic taus with one neutral pion to make a clean case

	if(tau->momentum().perp() < tauEtCut) return; // rtau visible only for boosted taus
	
	double rTau = 0;
	double ltrack = leadingPionMomentum(tau);
	double visibleTauE = visibleTauEnergy(tau);

	if(visibleTauE != 0) rTau = ltrack/visibleTauE;

	if(abs(mother) == 24) TauRtauW->Fill(rTau);
        if(abs(mother) == 37) TauRtauHpm->Fill(rTau); 
}

void TauValidation::spinEffects(const HepMC::GenParticle* tau,int mother, int decay){

	if(decay != pi) return; // polarization only for 1-prong hadronic taus with no neutral pions

	TLorentzVector momP4 = motherP4(tau);
	TLorentzVector pionP4 = leadingPionP4(tau);

	pionP4.Boost(-1*momP4.BoostVector());

	double energy = pionP4.E()/(momP4.M()/2);

	if(abs(mother) == 24) TauSpinEffectsW->Fill(energy);	
	if(abs(mother) == 37) TauSpinEffectsHpm->Fill(energy);
}

void TauValidation::spinEffectsZ(const HepMC::GenParticle* boson){

        TLorentzVector tautau(0,0,0,0);
	TLorentzVector pipi(0,0,0,0);

        int nSinglePionDecays = 0;
        if ( boson->end_vertex() ) {
              HepMC::GenVertex::particle_iterator des;
              for(des = boson->end_vertex()->particles_begin(HepMC::descendants);
                  des!= boson->end_vertex()->particles_end(HepMC::descendants);++des ) {

                        int pid = (*des)->pdg_id();
                        /*std::cout << " barcode=" << (*des)->barcode() << " pid="
                                  << pid << " mom=" << findMother(*des) << " status="
                                  << (*des)->status() << " px="
                                  << (*des)->momentum().px() << " decay=" 
                                  << findTauDecayChannel(*des) << std::endl;
			*/
                        if(abs(findMother(*des)) != 15 &&
                           abs(pid) == 15 && findTauDecayChannel(*des) == pi){
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
          TauSpinEffectsZ->Fill(pipi.M()/tautau.M());
        }
}

double TauValidation::leadingPionMomentum(const HepMC::GenParticle* tau){
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
                        //mother_pid = (*mother)->pdg_id();
                        //std::cout << " parent " << mother_pid << std::endl;
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

void TauValidation::photons(const HepMC::GenParticle* tau){

        if ( tau->end_vertex() ) {
	      double photonFromTauPtSum = 0;
	      bool photonFromTau = false;
              HepMC::GenVertex::particle_iterator des;
              for(des = tau->end_vertex()->particles_begin(HepMC::descendants);
                  des!= tau->end_vertex()->particles_end(HepMC::descendants);++des ) {
                        int pid = (*des)->pdg_id();
			if(pid == 22) {
				photonFromTauPtSum += (*des)->momentum().perp();
				photonFromTau = true;
			} 
              }
	      
	      TauPhotonsN->Fill(0.5);
	      TauPhotonsPt->Fill(0.5,tau->momentum().perp());
	      if(photonFromTau) {
		TauPhotonsN->Fill(1.5);
		TauPhotonsPt->Fill(1.5,photonFromTauPtSum);
	      }
        }
}

