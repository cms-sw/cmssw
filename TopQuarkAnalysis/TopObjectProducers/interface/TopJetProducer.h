//
// Author:  Jan Heyninck
// Created: Tue Apr  10 12:01:49 CEST 2007
//
// $Id: TopJetProducer.h,v 1.3 2007/06/10 08:57:28 lowette Exp $
//

#ifndef TopJetProducer_h
#define TopJetProducer_h

/**
  \class    TopJetProducer TopJetProducer.h "TopQuarkAnalysis/TopObjectProducers/interface/TopJetProducer.h"
  \brief    Produces TopJet's

   TopJetProducer produces TopJet's starting from a JetType collection,
   with possible adding of resolutions and more things to come

  \author   Jan Heyninck
  \version  $Id: TopJetProducer.h,v 1.3 2007/06/10 08:57:28 lowette Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "PhysicsTools/Utilities/interface/EtComparator.h"

#include "AnalysisDataFormats/TopObjects/interface/TopJet.h"
#include "DataFormats/BTauReco/interface/JetTag.h"

#include "DataFormats/BTauReco/interface/TrackProbabilityTagInfo.h"
#include "DataFormats/BTauReco/interface/TrackProbabilityTagInfoFwd.h"
#include "DataFormats/BTauReco/interface/TrackCountingTagInfo.h"
#include "DataFormats/BTauReco/interface/TrackCountingTagInfoFwd.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfoFwd.h"
#include "DataFormats/BTauReco/interface/TrackCountingTagInfoFwd.h"
#include "RecoBTag/MCTools/interface/JetFlavourIdentifier.h"

using namespace std;
using namespace edm;



class TopObjectResolutionCalc;


class TopJetProducer : public edm::EDProducer {

  public:

    explicit TopJetProducer(const edm::ParameterSet & iConfig);
    ~TopJetProducer();

    virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

  private:

    JetFlavourIdentifier jfi;
    // configurables
    std::string   jetTagsLabel_;
    edm::InputTag recJetsLabel_;
    edm::InputTag caliJetsLabel_;
    bool          storeBDiscriminants;
    bool          addResolutions_;
    std::string   caliJetResoFile_;
    bool          dropTrackCountingFromAOD   ;
    bool          dropTrackProbabilityFromAOD ;
    bool          dropSoftMuonFromAOD   ;      
    bool          dropSoftElectronFromAOD   ; 
    bool          keepdiscriminators; 
    bool          keepjettagref;



    // tools
    TopObjectResolutionCalc *   theResoCalc_;
    EtInverseComparator<TopJet> eTComparator_;

};


#endif
