#ifndef TauValidation_H
#define TauValidation_H

/*class TauValidation
 *  
 *  Class to fill Event Generator dqm monitor elements; works on HepMCProduct
 *
 *  $Date: 2012/11/03 12:05:14 $
 *  $Revision: 1.15 $
 *
 */

// framework & common header files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "TLorentzVector.h"

#include "Validation/EventGenerator/interface/WeightManager.h"

class TauValidation : public edm::EDAnalyzer
{
    public:
	// tau decays
        enum  {undetermined,
               electron,
               muon,
               pi,
               rho,
	       a1,
               K,
	       Kstar,
	       pi1pi0,
               pinpi0,
               tripi,
               tripinpi0,
	       stable};
	// tau mother particles 
	enum  {other,
	       B,
	       D,
	       gamma,
	       Z,
	       W,
	       HSM,
	       H0,
	       A0,
	       Hpm};

    public:
	explicit TauValidation(const edm::ParameterSet&);
	virtual ~TauValidation();
	virtual void beginJob();
	virtual void endJob();  
	virtual void analyze(const edm::Event&, const edm::EventSetup&);
	virtual void beginRun(const edm::Run&, const edm::EventSetup&);
	virtual void endRun(const edm::Run&, const edm::EventSetup&);

    private:
	int tauMother(const HepMC::GenParticle*, double weight);
	int tauProngs(const HepMC::GenParticle*, double weight);
	int tauDecayChannel(const HepMC::GenParticle*, double weight=0.0);
	int findMother(const HepMC::GenParticle*);
	bool isLastTauinChain(const HepMC::GenParticle* tau);
	void rtau(const HepMC::GenParticle*,int,int, double weight);
	void spinEffects(const HepMC::GenParticle*,int,int, double weight);
	void spinEffectsZ(const HepMC::GenParticle*, double weight);
	double leadingPionMomentum(const HepMC::GenParticle*, double weight);
	double visibleTauEnergy(const HepMC::GenParticle*);
	TLorentzVector leadingPionP4(const HepMC::GenParticle*);
	TLorentzVector motherP4(const HepMC::GenParticle*);
	void photons(const HepMC::GenParticle*, double weight);
	void findTauList(const HepMC::GenParticle* tau,std::vector<const HepMC::GenParticle*> &TauList);
	void findFSRandBrem(const HepMC::GenParticle* p, bool doBrem, std::vector<const HepMC::GenParticle*> &ListofFSR,
			   std::vector<const HepMC::GenParticle*> &ListofBrem);
	void FindPhotosFSR(const HepMC::GenParticle* p,std::vector<const HepMC::GenParticle*> &ListofFSR,double &BosonScale);
	const HepMC::GenParticle* GetMother(const HepMC::GenParticle* tau);

        WeightManager _wmanager;

    	edm::InputTag hepmcCollection_;

	double tauEtCut;

  	/// PDT table
  	edm::ESHandle<HepPDT::ParticleDataTable> fPDGTable ;
  
  	///ME's "container"
  	DQMStore *dbe;

        MonitorElement *nTaus, *nPrimeTaus;
  	MonitorElement *TauPt, *TauEta, *TauPhi, *TauProngs, *TauDecayChannels, *TauMothers, 
	               *TauRtauW, *TauRtauHpm,
	               *TauSpinEffectsW, *TauSpinEffectsHpm, *TauSpinEffectsZ,
	  *TauBremPhotonsN,*TauBremPhotonsPt,*TauBremPhotonsPtSum,*TauFSRPhotonsN,*TauFSRPhotonsPt,*TauFSRPhotonsPtSum;
	unsigned int NJAKID;
	MonitorElement *JAKID;
	std::vector<std::vector<MonitorElement *> > JAKInvMass;

};

#endif

