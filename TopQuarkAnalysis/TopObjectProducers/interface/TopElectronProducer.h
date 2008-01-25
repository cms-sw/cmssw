//
// $Id: TopElectronProducer.h,v 1.20 2007/12/14 13:52:01 jlamb Exp $
//

#ifndef TopObjectProducers_TopElectronProducer_h
#define TopObjectProducers_TopElectronProducer_h

/**
  \class    TopElectronProducer TopElectronProducer.h "TopQuarkAnalysis/TopObjectProducers/interface/TopElectronProducer.h"
  \brief    Produces TopElectron's

   TopElectronProducer produces TopElectron's starting from an ElectronType
   collection, with possible matching to generator level, adding of resolutions
   and calculation of a lepton likelihood ratio

  \author   Jan Heyninck, Steven Lowette
  \version  $Id: TopElectronProducer.h,v 1.20 2007/12/14 13:52:01 jlamb Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "PhysicsTools/Utilities/interface/PtComparator.h"
#include "AnalysisDataFormats/Egamma/interface/ElectronID.h"
#include "AnalysisDataFormats/Egamma/interface/ElectronIDAssociation.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/EgammaCandidates/interface/PMGsfElectronIsoCollection.h"
#include "DataFormats/EgammaCandidates/interface/PMGsfElectronIsoNumCollection.h"
#include "DataFormats/Candidate/interface/CandAssociation.h"
#include "DataFormats/Common/interface/View.h"

#include "AnalysisDataFormats/TopObjects/interface/TopElectron.h"

#include <string>


class TopObjectResolutionCalc;
class TrackerIsolationPt;
class CaloIsolationEnergy;
class TopLeptonLRCalc;


class TopElectronProducer : public edm::EDProducer {
  
  public:

    explicit TopElectronProducer(const edm::ParameterSet & iConfig);
    ~TopElectronProducer();  

    virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

  private:

    void removeGhosts(std::vector<TopElectronType> & elecs);
    reco::GenParticle findTruth(const reco::GenParticleCollection & parts, const TopElectronType & elec);
    void matchTruth(const reco::GenParticleCollection & particles, std::vector<TopElectronType> & electrons);
    double electronID(const edm::Handle<std::vector<TopElectronType> > & elecs, 
                      const edm::Handle<reco::ElectronIDAssociationCollection> & elecIDs, int idx);
    void setEgammaIso(TopElectron &anElectron,
		      const edm::Handle<std::vector<TopElectronType> > & elecs,
		      const edm::Handle<reco::PMGsfElectronIsoCollection> tkIsoHandle,
		      const edm::Handle<reco::PMGsfElectronIsoNumCollection> tkNumIsoHandle,
		      const edm::Handle<reco::CandViewDoubleAssociations> ecalIsoHandle,
		      const edm::Handle<reco::CandViewDoubleAssociations> hcalIsoHandle,
		      int idx);

  private:

    void removeEleDupes(std::vector<TopElectron> *electrons);

    // configurables
    edm::InputTag electronSrc_;
    bool          doGhostRemoval_;
    bool          addGenMatch_;
    edm::InputTag genPartSrc_;
    double        maxDeltaR_;
    double        minRecoOnGenEt_;
    double        maxRecoOnGenEt_;
    bool          addResolutions_;
    bool          useNNReso_;
    std::string   electronResoFile_;
    bool          addTrkIso_;
    edm::InputTag tracksSrc_;
    bool          addCalIso_;
    edm::InputTag towerSrc_;
    bool          addElecID_;
    edm::InputTag elecIDSrc_;
    bool          addElecIDRobust_;
    edm::InputTag elecIDRobustSrc_;
    bool          addLRValues_;
    std::string   electronLRFile_;
    bool          addEgammaIso_;
    edm::InputTag egammaTkIsoSrc_;
    edm::InputTag egammaTkNumIsoSrc_;
    edm::InputTag egammaEcalIsoSrc_;
    edm::InputTag egammaHcalIsoSrc_;

    // tools
    TopObjectResolutionCalc      * theResoCalc_;
    TrackerIsolationPt           * trkIsolation_;
    CaloIsolationEnergy          * calIsolation_;
    TopLeptonLRCalc              * theLeptonLRCalc_;
    GreaterByPt<TopElectron>       pTComparator_;
    // other
    std::vector<std::pair<const reco::GenParticle *, TopElectronType *> > pairGenRecoElectronsVector_;

};


#endif
