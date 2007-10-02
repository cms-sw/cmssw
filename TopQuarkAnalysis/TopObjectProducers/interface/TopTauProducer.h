//
// Author:  Christophe Delaere
// Created: Thu Jul  26 10:34:00 CEST 2007
//
// $Id: TopTauProducer.h,v 1.6 2007/10/01 20:58:02 lowette Exp $
//

#ifndef TopObjectProducers_TopTauProducer_h
#define TopObjectProducers_TopTauProducer_h

/**
  \class    TopTauProducer TopTauProducer.h "TopQuarkAnalysis/TopObjectProducers/interface/TopTauProducer.h"
  \brief    Produces TopTau's

   TopTauProducer produces TopTau's starting from a TauType collection,
   with possible matching to generator level, adding of resolutions and
   calculation of a lepton likelihood ratio

  \author   Christophe Delaere
  \version  $Id: TopTauProducer.h,v 1.6 2007/10/01 20:58:02 lowette Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "AnalysisDataFormats/TopObjects/interface/TopTau.h"
#include "PhysicsTools/Utilities/interface/PtComparator.h"

#include <string>

class TopObjectResolutionCalc;
class TopLeptonLRCalc;


class TopTauProducer : public edm::EDProducer {

  public:

    explicit TopTauProducer(const edm::ParameterSet & iConfig);
    ~TopTauProducer();

    virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

  private:

    // configurables
    edm::InputTag  tauSrc_;
    bool           addGenMatch_;
    edm::InputTag  genPartSrc_;
    bool           addResolutions_;
    bool           useNNReso_;
    std::string    tauResoFile_;
    bool           addLRValues_;
    std::string    tauLRFile_;
    bool           redoDiscriminant_;
    double         Rmatch_, Rsig_, Riso_, pT_LT_, pT_min_;
    // tools
    TopObjectResolutionCalc * theResoCalc_;
    TopLeptonLRCalc *         theLeptonLRCalc_;
    GreaterByPt<TopTau>       pTTauComparator_;
};


#endif
