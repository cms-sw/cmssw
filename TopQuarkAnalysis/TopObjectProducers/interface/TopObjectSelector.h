//
// Author:  Steven Lowette
// Created: Thu Jun  7 05:49:16 2007 UTC
//
// $Id: TopObjectSelector.h,v 1.3 2007/06/23 07:31:51 lowette Exp $
//

#ifndef TopObjectProducer_TopObjectSelector_h
#define TopObjectProducer_TopObjectSelector_h


#include "PhysicsTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleElementCollectionSelector.h"

#include "AnalysisDataFormats/TopObjects/interface/TopLepton.h"
#include "AnalysisDataFormats/TopObjects/interface/TopJet.h"
#include "AnalysisDataFormats/TopObjects/interface/TopMET.h"
#include "AnalysisDataFormats/TopObjects/interface/TopParticle.h"

#include <vector>


typedef ObjectSelector<
          SingleElementCollectionSelector<
            std::vector<TopJetType>,
            SingleObjectSelector<TopJetType>
          >
        > CaloJetSelector;
typedef ObjectSelector<
          SingleElementCollectionSelector<
            std::vector<TopElectron>,
            SingleObjectSelector<TopElectron>
          >
        > TopElectronSelector;
typedef ObjectSelector<
          SingleElementCollectionSelector<
            std::vector<TopTau>,
            SingleObjectSelector<TopTau>
          >
        > TopTauSelector;
typedef ObjectSelector<
          SingleElementCollectionSelector<
            std::vector<TopMuon>,
            SingleObjectSelector<TopMuon>
          >
        > TopMuonSelector;
typedef ObjectSelector<
          SingleElementCollectionSelector<
            std::vector<TopJet>,
            SingleObjectSelector<TopJet>
          >
        > TopJetSelector;
typedef ObjectSelector<
          SingleElementCollectionSelector<
            std::vector<TopMET>,
            SingleObjectSelector<TopMET>
          >
        > TopMETSelector;
typedef ObjectSelector<
          SingleElementCollectionSelector<
            std::vector<TopParticle>,
            SingleObjectSelector<TopParticle>
          >
        > TopParticleSelector;


#endif
