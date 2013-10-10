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


//#include <DQMServices/Core/interface/DQMStore.h>

#include <Validation/L1T/interface/L1ValidatorHists.h>

//
// class declaration
//

class L1Validator : public edm::EDAnalyzer {
  public:
    explicit L1Validator(const edm::ParameterSet&);
    ~L1Validator();

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


  private:
    //virtual void beginJob() override;
    //virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
    virtual void endJob() override;

    //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
    //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
    //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
    //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
    void analyze(const edm::Event&, const edm::EventSetup&);

    // ----------member data ---------------------------
    //DQMStore* _dbe;
    std::string _dirName;
    std::string _fileName;

    L1ValidatorHists *_Hists;

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
