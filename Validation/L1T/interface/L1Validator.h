#ifndef L1Validator_h
#define L1Validator_h

// -*- C++ -*-
//
// Package:    L1T
// Class:      L1Validator
// 
/**\class L1T L1Validator.cc Validation/L1T/plugins/L1Validator.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Scott Wilbur
//         Created:  Wed, 28 Aug 2013 09:42:55 GMT
// $Id$
//
//


// system include files
#include <memory>
#include <iostream>
#include <string>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <DataFormats/HepMCCandidate/interface/GenParticle.h>
#include <DataFormats/HepMCCandidate/interface/GenParticleFwd.h>
#include <DataFormats/L1Trigger/interface/L1EmParticle.h>
#include <DataFormats/L1Trigger/interface/L1EmParticleFwd.h>
#include <DataFormats/L1Trigger/interface/L1JetParticle.h>
#include <DataFormats/L1Trigger/interface/L1JetParticleFwd.h>
#include <DataFormats/L1Trigger/interface/L1MuonParticle.h>
#include <DataFormats/L1Trigger/interface/L1MuonParticleFwd.h>
#include <DataFormats/L1Trigger/interface/L1EtMissParticle.h>
#include <DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h>

#include <DQMServices/Core/interface/MonitorElement.h>
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

#include <Validation/L1T/interface/L1ValidatorHists.h>

//
// class declaration
//

class L1Validator : public DQMEDAnalyzer {
  public:
    explicit L1Validator(const edm::ParameterSet&);
    ~L1Validator();

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
    void analyze(const edm::Event&, const edm::EventSetup&) override;

  protected:
    void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

  private:
    // ----------member data ---------------------------
    std::string _dirName;
    std::string _fileName;

    edm::EDGetTokenT<reco::GenParticleCollection> _GenSource;
    edm::EDGetTokenT<l1extra::L1EmParticleCollection> _L1ExtraIsoEGSource;
    edm::EDGetTokenT<l1extra::L1EmParticleCollection> _L1ExtraNonIsoEGSource;
    edm::EDGetTokenT<l1extra::L1JetParticleCollection> _L1ExtraCenJetSource;
    edm::EDGetTokenT<l1extra::L1JetParticleCollection> _L1ExtraForJetSource;
    edm::EDGetTokenT<l1extra::L1JetParticleCollection> _L1ExtraTauJetSource;
    edm::EDGetTokenT<l1extra::L1MuonParticleCollection> _L1ExtraMuonSource;
    //edm::EDGetTokenT<l1extra::L1EtMissParticleCollection> _L1ExtraMETSource;

    L1ValidatorHists _Hists;

    //---------------helper functions------------------
  private:
    const reco::LeafCandidate *FindBest(const reco::GenParticle *, const std::vector<l1extra::L1EmParticle> *, const std::vector<l1extra::L1EmParticle> *);
    const reco::LeafCandidate *FindBest(const reco::GenParticle *, const std::vector<l1extra::L1JetParticle> *, const std::vector<l1extra::L1JetParticle> *);
    const reco::LeafCandidate *FindBest(const reco::GenParticle *, const std::vector<l1extra::L1MuonParticle> *);
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//

#endif
