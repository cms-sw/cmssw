// -*- C++ -*-
//
// Package:    ObjectAnalyzer
// Class:      TTbar_GenJetAnalyzer
//
/**\class TTbar_GenJetAnalyzer 

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Martijn Gosselink,,,
//         Created:  Thu May 10 17:15:16 CEST 2012
//
//
// Added to: Validation/EventGenerator by Ian M. Nugent June 28, 2012

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

#include <map>
#include <string>

//
// class declaration
//

class TTbar_GenJetAnalyzer : public DQMEDAnalyzer {
public:
  explicit TTbar_GenJetAnalyzer(const edm::ParameterSet &);
  ~TTbar_GenJetAnalyzer() override;

  void bookHistograms(DQMStore::IBooker &i, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;

private:
  // ----------member data ---------------------------

  edm::InputTag jets_;
  edm::InputTag genEventInfoProductTag_;
  std::map<std::string, MonitorElement *> hists_;

  double weight;

  edm::EDGetTokenT<GenEventInfoProduct> genEventInfoProductTagToken_;
  edm::EDGetTokenT<std::vector<reco::GenJet> > jetsToken_;
};
