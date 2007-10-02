//
// Author:  Steven Lowette
// Created: Tue Oct  2 19:55:00 CEST 2007
//
// $Id$
//

#ifndef TopObjectProducer_TopObjectFilter_h
#define TopObjectProducer_TopObjectFilter_h


#include "PhysicsTools/UtilAlgos/interface/ObjectCountFilter.h"

#include "AnalysisDataFormats/TopObjects/interface/TopElectron.h"
#include "AnalysisDataFormats/TopObjects/interface/TopMuon.h"
#include "AnalysisDataFormats/TopObjects/interface/TopTau.h"
#include "AnalysisDataFormats/TopObjects/interface/TopJet.h"
#include "AnalysisDataFormats/TopObjects/interface/TopMET.h"
#include "AnalysisDataFormats/TopObjects/interface/TopParticle.h"

#include <vector>


typedef ObjectCountFilter<std::vector<TopElectron> > TopElectronCountFilter;
typedef ObjectCountFilter<std::vector<TopMuon> >     TopMuonCountFilter;
typedef ObjectCountFilter<std::vector<TopTau> >      TopTauCountFilter;
typedef ObjectCountFilter<std::vector<TopJet> >      TopJetCountFilter;
typedef ObjectCountFilter<std::vector<TopMET> >      TopMETCountFilter;
typedef ObjectCountFilter<std::vector<TopParticle> > TopParticleCountFilter;


#endif
