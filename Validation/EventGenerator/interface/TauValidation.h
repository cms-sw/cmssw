#ifndef TauValidation_H
#define TauValidation_H

/*class TauValidation
 *  
 *  Class to fill Event Generator dqm monitor elements; works on HepMCProduct
 *
 *  $Date: 2011/02/10 15:01:07 $
 *  $Revision: 1.6 $
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
//               tripinpi0,
	       stable};
	// tau mother particles 
	enum  {other,
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
	int tauMother(const HepMC::GenParticle*);
	int tauProngs(const HepMC::GenParticle*);
	int tauDecayChannel(const HepMC::GenParticle*);
	int findMother(const HepMC::GenParticle*);
	int findTauDecayChannel(const HepMC::GenParticle*);
	void rtau(const HepMC::GenParticle*,int,int);
	void spinEffects(const HepMC::GenParticle*,int,int);
	void spinEffectsZ(const HepMC::GenParticle*);
	double leadingPionMomentum(const HepMC::GenParticle*);
	double visibleTauEnergy(const HepMC::GenParticle*);
	TLorentzVector leadingPionP4(const HepMC::GenParticle*);
	TLorentzVector motherP4(const HepMC::GenParticle*);
	void photons(const HepMC::GenParticle*);

    	edm::InputTag hepmcCollection_;

	double tauEtCut;

  	/// PDT table
  	edm::ESHandle<HepPDT::ParticleDataTable> fPDGTable ;
  
  	///ME's "container"
  	DQMStore *dbe;

        MonitorElement *nEvt;
  	MonitorElement *TauPt, *TauEta, *TauProngs, *TauDecayChannels, *TauMothers, 
                       *TauRtauW, *TauRtauHpm,
                       *TauSpinEffectsW, *TauSpinEffectsHpm, *TauSpinEffectsZ,
	               *TauPhotonsN,*TauPhotonsPt;
};

#endif

