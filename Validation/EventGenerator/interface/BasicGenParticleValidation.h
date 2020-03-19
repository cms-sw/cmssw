#ifndef BASICGENPARICLEVALIDATION_H
#define BASICGENPARICLEVALIDATION_H

/*class BasicGenParticleValidation
 *  
 *  Class to fill Event Generator dqm monitor elements; works on HepMCProduct
 *
 *
 */

// framework & common header files
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
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include "Validation/EventGenerator/interface/WeightManager.h"

class BasicGenParticleValidation : public DQMEDAnalyzer {
public:
  explicit BasicGenParticleValidation(const edm::ParameterSet &);
  ~BasicGenParticleValidation() override;

  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void bookHistograms(DQMStore::IBooker &i, edm::Run const &, edm::EventSetup const &) override;

  bool matchParticles(const HepMC::GenParticle *&, const reco::GenParticle *&);

private:
  WeightManager wmanager_;
  edm::InputTag hepmcCollection_;
  edm::InputTag genparticleCollection_;
  edm::InputTag genjetCollection_;
  double matchPr_;

  unsigned int verbosity_;

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

  edm::EDGetTokenT<edm::HepMCProduct> hepmcCollectionToken_;
  edm::EDGetTokenT<reco::GenParticleCollection> genparticleCollectionToken_;
  edm::EDGetTokenT<reco::GenJetCollection> genjetCollectionToken_;
};

#endif
