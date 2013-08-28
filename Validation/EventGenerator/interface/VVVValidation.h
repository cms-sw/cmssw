#ifndef VVVVALIDATION_H
#define VVVVALIDATION_H


// framework & common header files
#include "TLorentzVector.h" 
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "HepMC/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include "Validation/EventGenerator/interface/WeightManager.h"
class VVVValidation : public edm::EDAnalyzer
{
    public:
	explicit VVVValidation(const edm::ParameterSet&);
	virtual ~VVVValidation();
	virtual void beginJob();
	virtual void endJob();  
	virtual void analyze(const edm::Event&, const edm::EventSetup&);
	virtual void beginRun(const edm::Run&, const edm::EventSetup&);
	virtual void endRun(const edm::Run&, const edm::EventSetup&);

    bool matchParticles(const HepMC::GenParticle*&, const reco::GenParticle*&); 
    int getParentBarcode(HepMC::GenParticle* it);

    private:

    WeightManager _wmanager;

    edm::InputTag hepmcCollection_;
    edm::InputTag genparticleCollection_;
    edm::InputTag genjetCollection_;
    double matchPr_;
    double _lepStatus;
    double _motherStatus;	

    unsigned int verbosity_;

	/// PDT table
	edm::ESHandle<HepPDT::ParticleDataTable> fPDGTable ;
	
	///ME's "container"
	DQMStore *dbe;

    MonitorElement *nEvt;

    // Basic reco::GenParticle test
  
    // Basci GenJets analysis
    MonitorElement * mll;
    MonitorElement * ptll;
    MonitorElement * mlll;
    MonitorElement * ptlll;
    MonitorElement * mlllnununu;
    MonitorElement * mtlllnununu;
    MonitorElement * ptlllnununu;
    MonitorElement * leading_l_pt ;
    MonitorElement * subleading_l_pt ;
    MonitorElement * subsubleading_l_pt; 
    MonitorElement * leading_l_eta ;
    MonitorElement * subleading_l_eta ;
    MonitorElement * subsubleading_l_eta; 
    MonitorElement *genJetMult;
    MonitorElement *genJetEnergy;
    MonitorElement *genJetPt;
    MonitorElement *genJetEta;
    MonitorElement *genJetPhi;
    MonitorElement *genJetDeltaEtaMin;
    MonitorElement *h_dr;
    MonitorElement *genJetPto1;
    MonitorElement *genJetPto30;
    MonitorElement *genJetPto50;
    MonitorElement *genJetPto100;
    MonitorElement *genJetCentral;

    MonitorElement *genJetTotPt;
    MonitorElement *WW_TwoJEt_JetsM;

    MonitorElement *h_l_jet_eta;
    MonitorElement *h_l_jet_pt;
    MonitorElement *h_sl_jet_eta;
    MonitorElement *h_sl_jet_pt;
    MonitorElement *h_ssl_jet_eta;
    MonitorElement *h_ssl_jet_pt;

    MonitorElement *h_mWplus ;
    MonitorElement *h_phiWplus;
    MonitorElement *h_ptWplus ;
    MonitorElement *h_yWplus;

    MonitorElement *h_mWminus;
    MonitorElement *h_phiWminus;
    MonitorElement *h_ptWminus;
    MonitorElement *h_yWminus;

    MonitorElement *h_mZ;
    MonitorElement *h_phiZ;
    MonitorElement *h_ptZ;
    MonitorElement *h_yZ;
    MonitorElement *h_mWplus_3b ;
    MonitorElement *h_phiWplus_3b;
    MonitorElement *h_ptWplus_3b ;
    MonitorElement *h_yWplus_3b;

    MonitorElement *h_mWminus_3b;
    MonitorElement *h_phiWminus_3b;
    MonitorElement *h_ptWminus_3b;
    MonitorElement *h_yWminus_3b;

    MonitorElement *h_mZ_3b;
    MonitorElement *h_phiZ_3b;
    MonitorElement *h_ptZ_3b;
    MonitorElement *h_yZ_3b;

    MonitorElement *h_mWW;
    MonitorElement *h_phiWW;
    MonitorElement *h_ptWW;
    MonitorElement *h_yWW;

    MonitorElement *h_mWZ;
    MonitorElement *h_phiWZ;
    MonitorElement *h_ptWZ;
    MonitorElement *h_yWZ;

    MonitorElement *h_mZZ;
    MonitorElement *h_phiZZ;
    MonitorElement *h_ptZZ;
    MonitorElement *h_yZZ;

    MonitorElement *h_mWWW;
    MonitorElement *h_phiWWW;
    MonitorElement *h_ptWWW;
    MonitorElement *h_yWWW;

    MonitorElement *h_mWWZ;
    MonitorElement *h_phiWWZ;
    MonitorElement *h_ptWWZ;
    MonitorElement *h_yWWZ;

    MonitorElement *h_mWZZ;
    MonitorElement *h_phiWZZ;
    MonitorElement *h_ptWZZ;
    MonitorElement *h_yWZZ;

    MonitorElement *h_mZZZ;
    MonitorElement *h_phiZZZ;
    MonitorElement *h_ptZZZ;
    MonitorElement *h_yZZZ;


};

#endif
