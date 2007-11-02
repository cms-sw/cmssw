//
// Author:  Steven Lowette
// Created: Tue Oct  2 19:55:00 CEST 2007
//
// $Id: TopObjectFilter.h,v 1.1.4.1 2007/10/31 22:00:03 lowette Exp $
//

#ifndef TopObjectProducer_TopObjectFilter_h
#define TopObjectProducer_TopObjectFilter_h


#include "PhysicsTools/UtilAlgos/interface/AnySelector.h"
#include "PhysicsTools/UtilAlgos/interface/MinNumberSelector.h"
#include "TopQuarkAnalysis/TopObjectProducers/interface/MaxNumberSelector.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectCountFilter.h"

#include "AnalysisDataFormats/TopObjects/interface/TopElectron.h"
#include "AnalysisDataFormats/TopObjects/interface/TopMuon.h"
#include "AnalysisDataFormats/TopObjects/interface/TopTau.h"
#include "AnalysisDataFormats/TopObjects/interface/TopJet.h"
#include "AnalysisDataFormats/TopObjects/interface/TopMET.h"
#include "AnalysisDataFormats/TopObjects/interface/TopParticle.h"

#include <vector>


typedef ObjectCountFilter<std::vector<TopElectron>, AnySelector, MinNumberSelector> TopElectronMinFilter;
typedef ObjectCountFilter<std::vector<TopMuon>,     AnySelector, MinNumberSelector> TopMuonMinFilter;
typedef ObjectCountFilter<std::vector<TopTau>,      AnySelector, MinNumberSelector> TopTauMinFilter;
typedef ObjectCountFilter<std::vector<TopJet>,      AnySelector, MinNumberSelector> TopJetMinFilter;
typedef ObjectCountFilter<std::vector<TopMET>,      AnySelector, MinNumberSelector> TopMETMinFilter;
typedef ObjectCountFilter<std::vector<TopParticle>, AnySelector, MinNumberSelector> TopParticleMinFilter;

typedef ObjectCountFilter<std::vector<TopElectron>, AnySelector, MaxNumberSelector> TopElectronMaxFilter;
typedef ObjectCountFilter<std::vector<TopMuon>,     AnySelector, MaxNumberSelector> TopMuonMaxFilter;
typedef ObjectCountFilter<std::vector<TopTau>,      AnySelector, MaxNumberSelector> TopTauMaxFilter;
typedef ObjectCountFilter<std::vector<TopJet>,      AnySelector, MaxNumberSelector> TopJetMaxFilter;
typedef ObjectCountFilter<std::vector<TopMET>,      AnySelector, MaxNumberSelector> TopMETMaxFilter;
typedef ObjectCountFilter<std::vector<TopParticle>, AnySelector, MaxNumberSelector> TopParticleMaxFilter;


#endif
