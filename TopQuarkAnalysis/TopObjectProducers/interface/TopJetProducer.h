//
// $Id: TopJetProducer.h,v 1.15 2007/08/09 09:16:50 tsirig Exp $
//

#ifndef TopObjectProducers_TopJetProducer_h
#define TopObjectProducers_TopJetProducer_h

/**
  \class    TopJetProducer TopJetProducer.h "TopQuarkAnalysis/TopObjectProducers/interface/TopJetProducer.h"
  \brief    Produces TopJet's

   TopJetProducer produces TopJet's starting from a JetType collection,
   with possible adding of resolutions and more things to come

  \author   Jan Heyninck
  \version  $Id: TopJetProducer.h,v 1.15 2007/08/09 09:16:50 tsirig Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "PhysicsTools/Utilities/interface/EtComparator.h"

#include "AnalysisDataFormats/TopObjects/interface/TopJet.h"
#include "AnalysisDataFormats/TopObjects/interface/TopLepton.h"


#include "PhysicsTools/JetCharge/interface/JetCharge.h"
#include "TopQuarkAnalysis/TopObjectProducers/interface/SimpleJetTrackAssociator.h"

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
    bool          doJetCleaning_;
    edm::InputTag topElectronsLabel_;
    edm::InputTag topMuonsLabel_;
    float LEPJETDR_;
    float ELEISOCUT_;
    float MUISOCUT_;
    // TEMP End
    bool          getJetMCFlavour_;
    bool          doGenPartonMatch_;
    edm::InputTag genPartonSrc_;
    bool          doGenJetMatch_;
    edm::InputTag genJetSrc_;
    bool          doPartonJetMatch_;
    edm::InputTag partonJetSrc_;
    bool          addResolutions_;
    std::string   caliJetResoFile_;
    bool          storeBTagInfo_;
    bool          ignoreTrackCountingFromAOD_;
    bool          ignoreTrackProbabilityFromAOD_;
    bool          ignoreSoftMuonFromAOD_;      
    bool          ignoreSoftElectronFromAOD_; 
    bool          keepDiscriminators_; 
    bool          keepJetTagRefs_;
    bool          storeAssociatedTracks_;
    edm::ParameterSet trackAssociationPSet_;
    bool          computeJetCharge_;
    edm::ParameterSet jetChargePSet_;

    std::vector<std::string> tagModuleLabelsToIgnore_;
    std::vector<std::string> bTaggingTagInfoNames_;

    // tools
    TopObjectResolutionCalc *   theResoCalc_;
    JetFlavourIdentifier *      jetFlavId_;
    GreaterByEt<TopJet>         eTComparator_;
    JetCharge *                 jetCharge_;
    reco::helper::SimpleJetTrackAssociator    simpleJetTrackAssociator_;

};


#endif
