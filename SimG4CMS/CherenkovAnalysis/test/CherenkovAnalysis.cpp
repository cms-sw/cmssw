// -*- C++ -*-
//
// Package:    CherenkovAnalysis
// Class:      CherenkovAnalysis
// 
/**\class CherenkovAnalysis CherenkovAnalysis.cpp SimG4CMS/CherenkovAnalysis/test/CherenkovAnalysis.cpp

Description: <one line class summary>

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Frederic Ronga
//         Created:  Wed Mar 12 17:39:55 CET 2008
// $Id: CherenkovAnalysis.cpp,v 1.6 2010/02/11 00:14:56 wmtan Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include <TH1F.h>

class CherenkovAnalysis : public edm::EDAnalyzer {
public:
  explicit CherenkovAnalysis(const edm::ParameterSet&);
  ~CherenkovAnalysis() {}


private:
  edm::InputTag caloHitSource_;
  TH1F* hEnergy_;
  double maxEnergy_;
  int nBinsEnergy_;

  TH1F* hTimeStructure_;

  virtual void beginJob() {}
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() {}

};


//__________________________________________________________________________________________________
CherenkovAnalysis::CherenkovAnalysis(const edm::ParameterSet& iConfig) :
  caloHitSource_( iConfig.getParameter<edm::InputTag>("caloHitSource") ),
  maxEnergy_( iConfig.getParameter<double>("maxEnergy")),
  nBinsEnergy_( iConfig.getParameter<unsigned>("nBinsEnergy"))
{

  // Book histograms
  edm::Service<TFileService> tfile;

  if ( !tfile.isAvailable() )
    throw cms::Exception("BadConfig") << "TFileService unavailable: "
                                      << "please add it to config file";

  hEnergy_ = tfile->make<TH1F>("hEnergy","Total energy deposit [GeV]",
                               nBinsEnergy_,0,maxEnergy_);
  hTimeStructure_ = tfile->make<TH1F>("hTimeStructure","Time structure [ns]",
                                      100,0,0.3);

}


//__________________________________________________________________________________________________
void
CherenkovAnalysis::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<edm::PCaloHitContainer> pCaloHits;
  iEvent.getByLabel( caloHitSource_, pCaloHits );

  double totalEnergy = 0;

  // Loop on all hits and calculate total energy loss
  edm::PCaloHitContainer::const_iterator it    = pCaloHits.product()->begin();
  edm::PCaloHitContainer::const_iterator itend = pCaloHits.product()->end();
  for ( ; it != itend; ++it ) {
    totalEnergy += (*it).energy();
    hTimeStructure_->Fill( (*it).time(), (*it).energy() ); // Time weighted by energy...
//     edm::LogInfo("CherenkovAnalysis") << "Time = " << (*it).time() << std::endl;
  }
  
  edm::LogInfo("CherenkovAnalysis") << "Total energy = " << totalEnergy;
  hEnergy_->Fill( totalEnergy );

}


DEFINE_FWK_MODULE(CherenkovAnalysis);
