//
// Author:  Jan Heyninck
// Created: Tue Apr  10 12:01:49 CEST 2007
//
// $Id: TopJetProducer.h,v 1.10 2007/06/18 10:03:34 heyninck Exp $
//

#ifndef TopObjectProducers_TopJetProducer_h
#define TopObjectProducers_TopJetProducer_h

/**
  \class    TopJetProducer TopJetProducer.h "TopQuarkAnalysis/TopObjectProducers/interface/TopJetProducer.h"
  \brief    Produces TopJet's

   TopJetProducer produces TopJet's starting from a JetType collection,
   with possible adding of resolutions and more things to come

  \author   Jan Heyninck
  \version  $Id: TopJetProducer.h,v 1.10 2007/06/18 10:03:34 heyninck Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "PhysicsTools/Utilities/interface/EtComparator.h"

#include "AnalysisDataFormats/TopObjects/interface/TopJet.h"
#include "AnalysisDataFormats/TopObjects/interface/TopLepton.h"


class JetFlavourIdentifier;
class TopObjectResolutionCalc;


class TopJetProducer : public edm::EDProducer {

  public:

    explicit TopJetProducer(const edm::ParameterSet & iConfig);
    ~TopJetProducer();

    virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

  private:

    // TEMP Jet cleaning from electrons
    std::vector<TopElectron> selectIsolated(const std::vector<TopElectron> &electrons, float isoCut,
    							    const edm::EventSetup &iSetup, const edm::Event &iEvent);
    std::vector<TopMuon> selectIsolated(const std::vector<TopMuon> &muons, float isoCut,
    							    const edm::EventSetup &iSetup, const edm::Event &iEvent);
    // TEMP End

    // configurables
    edm::InputTag recJetsLabel_;
    edm::InputTag caliJetsLabel_;
    // TEMP Jet cleaning from electrons
    edm::InputTag topElectronsLabel_;
    edm::InputTag topMuonsLabel_;
    bool          doJetCleaning_;
    // TEMP End
    bool          addResolutions_;
    std::string   caliJetResoFile_;
    bool          storeBTagInfo_;
    bool          ignoreTrackCountingFromAOD_;
    bool          ignoreTrackProbabilityFromAOD_;
    bool          ignoreSoftMuonFromAOD_;      
    bool          ignoreSoftElectronFromAOD_; 
    bool          keepDiscriminators_; 
    bool          keepJetTagRefs_;
    bool          getJetMCFlavour_;

    // TEMP Jet cleaning from electrons
    float LEPJETDR_;
    float ELEISOCUT_;
    float MUISOCUT_;
    // TEMP End

    // tools
    TopObjectResolutionCalc *   theResoCalc_;
    JetFlavourIdentifier *      jetFlavId_;
    EtInverseComparator<TopJet> eTComparator_;

};


#endif
