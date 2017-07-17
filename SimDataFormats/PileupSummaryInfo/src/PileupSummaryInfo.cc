// -*- C++ -*-
//
// Package:     PileupSummaryInfo
// Class  :     PileupSummaryInfo
// 
// Implementation:
//
//
// Original Author:  Mike Hildreth, Notre Dame
//         Created:  
//

#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"

PileupSummaryInfo::PileupSummaryInfo( const int num_PU_vertices,
                    const std::vector<float>& zpositions, 
                    const std::vector<float>& times, 
                    const std::vector<float>& sumpT_lowpT,
                    const std::vector<float>& sumpT_highpT,
                    const std::vector<int>&   ntrks_lowpT,
                    const std::vector<int>&   ntrks_highpT ) :
    num_PU_vertices_(num_PU_vertices),
    zpositions_(zpositions),
    times_(times),
    sumpT_lowpT_(sumpT_lowpT),
    sumpT_highpT_(sumpT_highpT),
    ntrks_lowpT_(ntrks_lowpT),
    ntrks_highpT_(ntrks_highpT)
{
}

PileupSummaryInfo::PileupSummaryInfo( const int num_PU_vertices,
                     const std::vector<float>& zpositions, 
                     const std::vector<float>& times, 
                     const std::vector<float>& sumpT_lowpT,
                     const std::vector<float>& sumpT_highpT,
                     const std::vector<int>&   ntrks_lowpT,
		     const std::vector<int>&   ntrks_highpT,
		     int bunchCrossing) :
    num_PU_vertices_(num_PU_vertices),
    zpositions_(zpositions),
    times_(times),
    sumpT_lowpT_(sumpT_lowpT),
    sumpT_highpT_(sumpT_highpT),
    ntrks_lowpT_(ntrks_lowpT),
    ntrks_highpT_(ntrks_highpT),
    bunchCrossing_(bunchCrossing)
{
}


PileupSummaryInfo::PileupSummaryInfo( const int num_PU_vertices,
                     const std::vector<float>& zpositions, 
                     const std::vector<float>& times, 
                     const std::vector<float>& sumpT_lowpT,
                     const std::vector<float>& sumpT_highpT,
                     const std::vector<int>&   ntrks_lowpT,
		     const std::vector<int>&   ntrks_highpT,
		     const std::vector<edm::EventID>& eventInfo,
                     const std::vector<float>& pThats, 
		     int bunchCrossing,
		     float TrueNumInteractions,
 	             int bunchSpacing):
  num_PU_vertices_(num_PU_vertices),
  zpositions_(zpositions),
  times_(times),
  sumpT_lowpT_(sumpT_lowpT),
  sumpT_highpT_(sumpT_highpT),
  ntrks_lowpT_(ntrks_lowpT),
  ntrks_highpT_(ntrks_highpT),
  eventInfo_(eventInfo),
  pT_hats_(pThats),
  bunchCrossing_(bunchCrossing),
  bunchSpacing_(bunchSpacing),
  TrueNumInteractions_(TrueNumInteractions)
{
}



PileupSummaryInfo::PileupSummaryInfo( const int num_PU_vertices,
				      const std::vector<float>& instLumi,
				      const std::vector<edm::EventID>& eventInfo) :
    num_PU_vertices_(num_PU_vertices),
    eventInfo_(eventInfo),
    instLumi_(instLumi)
{
}

PileupSummaryInfo::~PileupSummaryInfo(){
}
