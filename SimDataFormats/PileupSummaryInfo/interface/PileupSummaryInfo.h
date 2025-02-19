#ifndef PileupSummaryInfo_h
#define PileupSummaryInfo_h
// -*- C++ -*-
//
// Package:     PileupSummaryInfo
// Class  :     PileupSummaryInfo
// 
/**\class PileupSummaryInfo PileupSummaryInfo.h SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h

Description: contains information related to the details of the pileup simulation for a given event
Usage: purely descriptive
*/
//
// Original Author:  Mike Hildreth, Notre Dame
//         Created:  July 1, 2010
// $Id: PileupSummaryInfo.h,v 1.4 2011/07/05 00:40:39 mikeh Exp $
//

#include "DataFormats/Provenance/interface/EventID.h"
#include <vector>
#include <string>



class PileupSummaryInfo {

 public:

  PileupSummaryInfo(){};

  PileupSummaryInfo( const int num_PU_vertices,
		     std::vector<float>& zpositions,
		     std::vector<float>& sumpT_lowpT,
		     std::vector<float>& sumpT_highpT,
		     std::vector<int>& ntrks_lowpT,
		     std::vector<int>& ntrks_highpT );

  PileupSummaryInfo( const int num_PU_vertices,
		     std::vector<float>& zpositions,
		     std::vector<float>& sumpT_lowpT,
		     std::vector<float>& sumpT_highpT,
		     std::vector<int>& ntrks_lowpT,
		     std::vector<int>& ntrks_highpT,
		     int bunchCrossing);


  PileupSummaryInfo( const int num_PU_vertices,
		     std::vector<float>& zpositions,
		     std::vector<float>& sumpT_lowpT,
		     std::vector<float>& sumpT_highpT,
		     std::vector<int>& ntrks_lowpT,
		     std::vector<int>& ntrks_highpT,
		     int bunchCrossing,
		     float TrueNumInteractions);

  PileupSummaryInfo( const int num_PU_vertices,
		     std::vector<float>& instLumi,
		     std::vector<edm::EventID>& eventInfo );


  ~PileupSummaryInfo();

  const int getPU_NumInteractions() const { return num_PU_vertices_; }
  const std::vector<float>& getPU_zpositions() const { return zpositions_; }
  const std::vector<float>& getPU_sumpT_lowpT() const { return sumpT_lowpT_; }
  const std::vector<float>& getPU_sumpT_highpT() const { return sumpT_highpT_; }
  const std::vector<int>& getPU_ntrks_lowpT() const { return ntrks_lowpT_; }
  const std::vector<int>& getPU_ntrks_highpT() const { return ntrks_highpT_; }
  const std::vector<float>& getPU_instLumi() const { return instLumi_; }
  const std::vector<edm::EventID>& getPU_EventID() const { return eventInfo_; }
  const int getBunchCrossing() const { return bunchCrossing_;}
  const float getTrueNumInteractions() const { return TrueNumInteractions_;}

 private:

  // for "standard" pileup: we have MC Truth information for these

  int num_PU_vertices_;
  std::vector<float> zpositions_;
  std::vector<float> sumpT_lowpT_;
  std::vector<float> sumpT_highpT_;
  std::vector<int> ntrks_lowpT_;
  std::vector<int> ntrks_highpT_;
  int bunchCrossing_;
  float TrueNumInteractions_;


  // for DataMixer pileup, we only have raw information:

  std::vector<float> instLumi_;
  std::vector<edm::EventID> eventInfo_;

};

#endif
