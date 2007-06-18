//
// Author:  Jan Heyninck
// Created: Tue Apr  10 12:01:49 CEST 2007
//
// $Id: TopJetProducer.h,v 1.9 2007/06/16 06:11:04 lowette Exp $
//

#ifndef TopJetProducer_h
#define TopJetProducer_h

/**
  \class    TopJetProducer TopJetProducer.h "TopQuarkAnalysis/TopObjectProducers/interface/TopJetProducer.h"
  \brief    Produces TopJet's

   TopJetProducer produces TopJet's starting from a JetType collection,
   with possible adding of resolutions and more things to come

  \author   Jan Heyninck
  \version  $Id: TopJetProducer.h,v 1.9 2007/06/16 06:11:04 lowette Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "PhysicsTools/Utilities/interface/EtComparator.h"

#include "AnalysisDataFormats/TopObjects/interface/TopJet.h"
#include "AnalysisDataFormats/TopObjects/interface/TopLepton.h"
#include "DataFormats/BTauReco/interface/JetTag.h"

#include "DataFormats/BTauReco/interface/TrackProbabilityTagInfo.h"
#include "DataFormats/BTauReco/interface/TrackProbabilityTagInfoFwd.h"
#include "DataFormats/BTauReco/interface/TrackCountingTagInfo.h"
#include "DataFormats/BTauReco/interface/TrackCountingTagInfoFwd.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfoFwd.h"
#include "DataFormats/BTauReco/interface/TrackCountingTagInfoFwd.h"
#include "RecoBTag/MCTools/interface/JetFlavourIdentifier.h"


class TopObjectResolutionCalc;


class TopJetProducer : public edm::EDProducer {

  public:

    explicit TopJetProducer(const edm::ParameterSet & iConfig);
    ~TopJetProducer();

    virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

  private:

    std::vector<TopElectron> selectIsolated(const std::vector<TopElectron> &electrons, float isoCut,
    							    const edm::EventSetup &iSetup, const edm::Event &iEvent);
    std::vector<TopMuon> selectIsolated(const std::vector<TopMuon> &muons, float isoCut,
    							    const edm::EventSetup &iSetup, const edm::Event &iEvent);


    // configurables
    edm::InputTag recJetsLabel_;
    edm::InputTag caliJetsLabel_;
    edm::InputTag jetTagsLabel_;
    edm::InputTag topElectronsLabel_;
    edm::InputTag topMuonsLabel_;
    bool          doJetCleaning_;
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

    float LEPJETDR_;
    float ELEISOCUT_;
    float MUISOCUT_;


    // tools
    TopObjectResolutionCalc *   theResoCalc_;
    JetFlavourIdentifier *      jetFlavId_;
    EtInverseComparator<TopJet> eTComparator_;

};


#endif
