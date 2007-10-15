//
// $Id: TopMuonProducer.h,v 1.16 2007/10/04 23:35:21 lowette Exp $
//

#ifndef TopObjectProducers_TopMuonProducer_h
#define TopObjectProducers_TopMuonProducer_h

/**
  \class    TopMuonProducer TopMuonProducer.h "TopQuarkAnalysis/TopObjectProducers/interface/TopMuonProducer.h"
  \brief    Produces TopMuon's

   TopMuonProducer produces TopMuon's starting from a MuonType collection,
   with possible matching to generator level, adding of resolutions and
   calculation of a lepton likelihood ratio

  \author   Jan Heyninck, Steven Lowette
  \version  $Id: TopMuonProducer.h,v 1.16 2007/10/04 23:35:21 lowette Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "PhysicsTools/Utilities/interface/PtComparator.h"

#include "AnalysisDataFormats/TopObjects/interface/TopMuon.h"

#include <string>


class TopObjectResolutionCalc;
class TopLeptonLRCalc;


class TopMuonProducer : public edm::EDProducer {
  
  public:
  
    explicit TopMuonProducer(const edm::ParameterSet & iConfig);
    ~TopMuonProducer();

    virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);
  
  private:

    reco::GenParticleCandidate findTruth(const reco::CandidateCollection & parts, const TopMuonType & muon);  
    void matchTruth(const reco::CandidateCollection & parts, std::vector<TopMuonType> & muons);
  
  private:
  
    // configurables
    edm::InputTag muonSrc_;
    bool          addGenMatch_;
    edm::InputTag genPartSrc_;
    double        maxDeltaR_;
    double        minRecoOnGenEt_;
    double        maxRecoOnGenEt_;
    bool          addResolutions_;
    bool          useNNReso_;
    std::string   muonResoFile_;
    bool          doTrkIso_;
    bool          doCalIso_;
    edm::InputTag trackIsoSrc_;
    edm::InputTag ecalIsoSrc_;
    edm::InputTag hcalIsoSrc_;
    edm::InputTag hocalIsoSrc_;
    bool          addMuonID_;
    bool          addLRValues_;
    edm::InputTag tracksSrc_;
    std::string   muonLRFile_;
    // tools
    TopObjectResolutionCalc * theResoCalc_;
    TopLeptonLRCalc         * theLeptonLRCalc_;
    GreaterByPt<TopMuon>      pTComparator_;
    // other
    std::vector<std::pair<const reco::Candidate *, TopMuonType *> > pairGenRecoMuonsVector_;

};


#endif
