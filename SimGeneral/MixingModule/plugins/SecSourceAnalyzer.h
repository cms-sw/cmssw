// -*- C++ -*-
//
// Package:    SecSourceAnalyzer
// Class:      SecSourceAnalyzer
// 
/**\class SecSourceAnalyzer SecSourceAnalyzer.cc SecSource/SecSourceAnalyzer/src/SecSourceAnalyzer.cc
*/
//
// Original Author:  Emilia Lubenova Becheva
//         Created:  Wed Apr 22 16:54:31 CEST 2009
//
//

#ifndef SecSourceAnalyzer_h
#define SecSourceAnalyzer_h

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Provenance/interface/EventID.h"

#include "Mixing/Base/interface/PileUp.h"


//
// class declaration
//
namespace edm {

  class ModuleCallingContext;

  class SecSourceAnalyzer : public edm::EDAnalyzer {
  public:
   
    explicit SecSourceAnalyzer(const edm::ParameterSet&);
    ~SecSourceAnalyzer();

    virtual void getBranches(EventPrincipal const& ep,
                             ModuleCallingContext const*);
    virtual void dummyFunction(EventPrincipal const& ep) {}

  private:
    virtual void beginJob() ;
    virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
    virtual void endJob() ;

    // ----------member data ---------------------------
    int minBunch_;
    int maxBunch_;

    bool dataStep2_;
    edm::InputTag label_;
      
    std::vector<std::vector<edm::SecondaryEventIDAndFileInfo> > vectorEventIDs_;

    std::shared_ptr<PileUp> input_;
    std::vector< float > TrueNumInteractions_[5];

    InputTag tag_;
 
  };
}//edm
#endif
