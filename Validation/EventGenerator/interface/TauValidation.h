#ifndef TauValidation_H
#define TauValidation_H

// framework & common header files
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
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "TLorentzVector.h"

class TauValidation : public DQMEDAnalyzer
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
	virtual void bookHistograms(DQMStore::IBooker &i, edm::Run const &, edm::EventSetup const &) override;
	virtual void dqmBeginRun(const edm::Run& r, const edm::EventSetup& c) override;
	virtual void analyze(edm::Event const&, edm::EventSetup const&) override;
    private:
	//	  WeightManager wmanager_;

	int tauMother(const reco::GenParticle*, double weight);
	int tauProngs(const reco::GenParticle*, double weight);
	int tauDecayChannel(const reco::GenParticle* tau,int jak_id,unsigned int TauBitMask, double weight);
	int findMother(const reco::GenParticle*);
	bool isLastTauinChain(const reco::GenParticle* tau);
	void spinEffectsWHpm(const reco::GenParticle*,int,int,std::vector<const reco::GenParticle*> &part,double weight);
	void spinEffectsZH(const reco::GenParticle* boson, double weight);
	double leadingPionMomentum(const reco::GenParticle*, double weight);
	double visibleTauEnergy(const reco::GenParticle*);
	TLorentzVector leadingPionP4(const reco::GenParticle*);
	TLorentzVector motherP4(const reco::GenParticle*);
	void photons(const reco::GenParticle*, double weight);
	void findTauList(const reco::GenParticle* tau,std::vector<const reco::GenParticle*> &TauList);
	void findFSRandBrem(const reco::GenParticle* p, bool doBrem, std::vector<const reco::GenParticle*> &ListofFSR,
			   std::vector<const reco::GenParticle*> &ListofBrem);
	void FindPhotosFSR(const reco::GenParticle* p,std::vector<const reco::GenParticle*> &ListofFSR,double &BosonScale);
	const reco::GenParticle* GetMother(const reco::GenParticle* tau);
	const std::vector<const reco::GenParticle*> GetMothers(const reco::GenParticle* boson);
	double Zstoa(double zs);
	void countParticles(const reco::GenParticle* p,int &allCount, int &eCount, int &muCount,
			    int &pi0Count,int &piCount,int &rhoCount,int &a1Count,int &KCount,int &KstarCount);

    	edm::InputTag genparticleCollection_;

  	/// PDT table
  	edm::ESHandle<HepPDT::ParticleDataTable> fPDGTable ;
  
        MonitorElement *nTaus, *nPrimeTaus;
  	MonitorElement *TauPt, *TauEta, *TauPhi, *TauProngs, *TauDecayChannels, *TauMothers, 
	  *TauSpinEffectsW_X, *TauSpinEffectsW_UpsilonRho, *TauSpinEffectsW_UpsilonA1,*TauSpinEffectsW_eX,*TauSpinEffectsW_muX,
	  *TauSpinEffectsHpm_X, *TauSpinEffectsHpm_UpsilonRho, *TauSpinEffectsHpm_UpsilonA1,*TauSpinEffectsHpm_eX,*TauSpinEffectsHpm_muX, 
	  *TauSpinEffectsZ_MVis, *TauSpinEffectsZ_Zs, *TauSpinEffectsZ_Xf, *TauSpinEffectsZ_Xb,
	  *TauSpinEffectsZ_X50to75,*TauSpinEffectsZ_X75to88,*TauSpinEffectsZ_X88to100,*TauSpinEffectsZ_X100to120,*TauSpinEffectsZ_X120UP,
	  *TauSpinEffectsZ_eX, *TauSpinEffectsZ_muX, *TauSpinEffectsZ_X, *TauSpinEffectsH_X,
	  *TauSpinEffectsH_MVis, *TauSpinEffectsH_Zs, *TauSpinEffectsH_Xf, *TauSpinEffectsH_Xb,
	  *TauSpinEffectsH_eX, *TauSpinEffectsH_muX, *TauSpinEffectsH_rhorhoAcoplanarityplus,  *TauSpinEffectsH_rhorhoAcoplanarityminus,
	  *TauBremPhotonsN,*TauBremPhotonsPt,*TauBremPhotonsPtSum,*TauFSRPhotonsN,*TauFSRPhotonsPt,*TauFSRPhotonsPtSum,
	  *TauSpinEffectsH_pipiAcoplanarity,*TauSpinEffectsH_pipiAcollinearity,*TauSpinEffectsH_pipiAcollinearityzoom, *DecayLength,
	  *LifeTime;

	unsigned int NMODEID;
	MonitorElement *MODEID;
	std::vector<std::vector<MonitorElement *> > MODEInvMass;

	int zsbins;
	double zsmin,zsmax;

	edm::EDGetTokenT<reco::GenParticleCollection> genparticleCollectionToken_;
};

#endif

