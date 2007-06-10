//
// Author:  Jan Heyninck
// Created: Tue Apr  10 12:01:49 CEST 2007
//
// $Id$
//

#ifndef TopJetProducer_h
#define TopJetProducer_h

/**
  \class    TopJetProducer TopJetProducer.h "TopQuarkAnalysis/TopObjectProducers/interface/TopJetProducer.h"
  \brief    Produces TopJet's

   TopJetProducer produces TopJet's starting from a JetType collection,
   with possible adding of resolutions and more things to come

  \author   Jan Heyninck
  \version  $Id$
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "PhysicsTools/Utilities/interface/EtComparator.h"

#include "AnalysisDataFormats/TopObjects/interface/TopJet.h"
#include "DataFormats/BTauReco/interface/JetTag.h"


class TopObjectResolutionCalc;


class TopJetProducer : public edm::EDProducer {

  public:

    explicit TopJetProducer(const edm::ParameterSet & iConfig);
    ~TopJetProducer();

    virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

  private:

    // configurables
    edm::InputTag jetTagsLabel_;
    edm::InputTag recJetsLabel_;
    edm::InputTag caliJetsLabel_;
    bool          addResolutions_;
    std::string   caliJetResoFile_;
    // tools
    TopObjectResolutionCalc *   theResoCalc_;
    EtInverseComparator<TopJet> eTComparator_;

};


#endif
