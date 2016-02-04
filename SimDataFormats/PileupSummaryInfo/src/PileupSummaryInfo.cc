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
// $Id: PileupSummaryInfo.cc,v 1.5 2011/07/05 00:40:41 mikeh Exp $
//

#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"


PileupSummaryInfo::PileupSummaryInfo( const int num_PU_vertices,
                     std::vector<float>& zpositions, 
                     std::vector<float>& sumpT_lowpT,
                     std::vector<float>& sumpT_highpT,
                     std::vector<int>&   ntrks_lowpT,
                     std::vector<int>&   ntrks_highpT )
{

  num_PU_vertices_ =  num_PU_vertices;
  zpositions_.clear();
  sumpT_lowpT_.clear();
  sumpT_highpT_.clear();
  ntrks_lowpT_.clear();
  ntrks_highpT_.clear();
  instLumi_.clear();
  eventInfo_.clear();

  int NLoop = zpositions.size();

  for( int ivtx = 0; ivtx<NLoop ; ++ivtx) {
    zpositions_.push_back(zpositions[ivtx]); 
    sumpT_lowpT_.push_back(sumpT_lowpT[ivtx]);
    sumpT_highpT_.push_back(sumpT_highpT[ivtx]);
    ntrks_lowpT_.push_back(ntrks_lowpT[ivtx]);
    ntrks_highpT_.push_back(ntrks_highpT[ivtx]);
  }

}

PileupSummaryInfo::PileupSummaryInfo( const int num_PU_vertices,
                     std::vector<float>& zpositions, 
                     std::vector<float>& sumpT_lowpT,
                     std::vector<float>& sumpT_highpT,
                     std::vector<int>&   ntrks_lowpT,
		     std::vector<int>&   ntrks_highpT,
		     int bunchCrossing)
{

  num_PU_vertices_ =  num_PU_vertices;
  zpositions_.clear();
  sumpT_lowpT_.clear();
  sumpT_highpT_.clear();
  ntrks_lowpT_.clear();
  ntrks_highpT_.clear();
  instLumi_.clear();
  eventInfo_.clear();
  bunchCrossing_ = bunchCrossing;

  int NLoop = zpositions.size();

  for( int ivtx = 0; ivtx<NLoop ; ++ivtx) {
    zpositions_.push_back(zpositions[ivtx]); 
    sumpT_lowpT_.push_back(sumpT_lowpT[ivtx]);
    sumpT_highpT_.push_back(sumpT_highpT[ivtx]);
    ntrks_lowpT_.push_back(ntrks_lowpT[ivtx]);
    ntrks_highpT_.push_back(ntrks_highpT[ivtx]);
  }

}


PileupSummaryInfo::PileupSummaryInfo( const int num_PU_vertices,
                     std::vector<float>& zpositions, 
                     std::vector<float>& sumpT_lowpT,
                     std::vector<float>& sumpT_highpT,
                     std::vector<int>&   ntrks_lowpT,
		     std::vector<int>&   ntrks_highpT,
		     int bunchCrossing,
		     float TrueNumInteractions )
{

  num_PU_vertices_ =  num_PU_vertices;
  zpositions_.clear();
  sumpT_lowpT_.clear();
  sumpT_highpT_.clear();
  ntrks_lowpT_.clear();
  ntrks_highpT_.clear();
  instLumi_.clear();
  eventInfo_.clear();
  bunchCrossing_ = bunchCrossing;
  TrueNumInteractions_ = TrueNumInteractions;

  int NLoop = zpositions.size();

  for( int ivtx = 0; ivtx<NLoop ; ++ivtx) {
    zpositions_.push_back(zpositions[ivtx]); 
    sumpT_lowpT_.push_back(sumpT_lowpT[ivtx]);
    sumpT_highpT_.push_back(sumpT_highpT[ivtx]);
    ntrks_lowpT_.push_back(ntrks_lowpT[ivtx]);
    ntrks_highpT_.push_back(ntrks_highpT[ivtx]);
  }

}



PileupSummaryInfo::PileupSummaryInfo( const int num_PU_vertices,
				      std::vector<float>& instLumi,
				      std::vector<edm::EventID>& eventInfo)
{
  num_PU_vertices_ =  num_PU_vertices;
  zpositions_.clear();
  sumpT_lowpT_.clear();
  sumpT_highpT_.clear();
  ntrks_lowpT_.clear();
  ntrks_highpT_.clear();
  instLumi_.clear();
  eventInfo_.clear();


  for( int ivtx = 0; ivtx<num_PU_vertices ; ++ivtx) {
    instLumi_.push_back(instLumi[ivtx]);
    eventInfo_.push_back(eventInfo[ivtx]);
  }
}

PileupSummaryInfo::~PileupSummaryInfo(){
}
