//
// Author:  Steven Lowette
// Created: Thu Jun  7 05:49:16 2007 UTC
//
// $Id: TopObjectSelector.h,v 1.5 2007/08/06 14:37:41 tsirig Exp $
//

#ifndef TopObjectProducer_TopObjectSelector_h
#define TopObjectProducer_TopObjectSelector_h


#include "PhysicsTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleElementCollectionSelector.h"

#include "AnalysisDataFormats/TopObjects/interface/TopElectron.h"
#include "AnalysisDataFormats/TopObjects/interface/TopMuon.h"
#include "AnalysisDataFormats/TopObjects/interface/TopTau.h"
#include "AnalysisDataFormats/TopObjects/interface/TopJet.h"
#include "AnalysisDataFormats/TopObjects/interface/TopMET.h"
#include "AnalysisDataFormats/TopObjects/interface/TopParticle.h"

#include <vector>


typedef SingleObjectSelector<
            std::vector<TopJetType>,
            StringCutObjectSelector<TopJetType>
        > CaloJetSelector;
typedef SingleObjectSelector<
            std::vector<TopElectron>,
            StringCutObjectSelector<TopElectron>
        > TopElectronSelector;
typedef SingleObjectSelector<
            std::vector<TopMuon>,
            StringCutObjectSelector<TopMuon>
        > TopMuonSelector;
typedef SingleObjectSelector<
            std::vector<TopTau>,
            StringCutObjectSelector<TopTau>
        > TopTauSelector;
typedef SingleObjectSelector<
            std::vector<TopJet>,
            StringCutObjectSelector<TopJet>
        > TopJetSelector;
typedef SingleObjectSelector<
            std::vector<TopMET>,
            StringCutObjectSelector<TopMET>
        > TopMETSelector;
typedef SingleObjectSelector<
            std::vector<TopParticle>,
            StringCutObjectSelector<TopParticle>
        > TopParticleSelector;


#endif
