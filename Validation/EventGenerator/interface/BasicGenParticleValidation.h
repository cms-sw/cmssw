#ifndef BASICGENPARICLEVALIDATION_H
#define BASICGENPARICLEVALIDATION_H

/*class BasicGenParticleValidation
 *  
 *  Class to fill Event Generator dqm monitor elements; works on HepMCProduct
 *
 *  $Date: 2011/12/29 10:53:10 $
 *  $Revision: 1.3 $
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

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include "Validation/EventGenerator/interface/WeightManager.h"

class BasicGenParticleValidation : public edm::EDAnalyzer
{
    public:
	explicit BasicGenParticleValidation(const edm::ParameterSet&);
	virtual ~BasicGenParticleValidation();
	virtual void beginJob();
	virtual void endJob();  
	virtual void analyze(const edm::Event&, const edm::EventSetup&);
	virtual void beginRun(const edm::Run&, const edm::EventSetup&);
	virtual void endRun(const edm::Run&, const edm::EventSetup&);

    bool matchParticles(const HepMC::GenParticle*&, const reco::GenParticle*&); 

    private:

    WeightManager _wmanager;

    edm::InputTag hepmcCollection_;
    edm::InputTag genparticleCollection_;
    edm::InputTag genjetCollection_;
    double matchPr_;	

    unsigned int verbosity_;

	/// PDT table
	edm::ESHandle<HepPDT::ParticleDataTable> fPDGTable ;
	
	///ME's "container"
	DQMStore *dbe;

    MonitorElement *nEvt;

    // Basic reco::GenParticle test
  
    MonitorElement *genPMultiplicity;
    MonitorElement *genMatched;
    MonitorElement *multipleMatching;
    MonitorElement *matchedResolution;

    // Basci GenJets analysis

    MonitorElement *genJetMult;
    MonitorElement *genJetEnergy;
    MonitorElement *genJetPt;
    MonitorElement *genJetEta;
    MonitorElement *genJetPhi;
    MonitorElement *genJetDeltaEtaMin;

    MonitorElement *genJetPto1;
    MonitorElement *genJetPto10;
    MonitorElement *genJetPto100;
    MonitorElement *genJetCentral;

    MonitorElement *genJetTotPt;

};

#endif
