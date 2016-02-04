#ifndef PileupMixingContent_h
#define PileupMixingContent_h
// -*- C++ -*-
//
// Package:     PileupMixingContent
// Class  :     PileupMixingContent
// 
/**\class PileupMixingContent PileupMixingContent.h SimDataFormats/PileupMixingContent/interface/PileupMixingContent.h

Description: contains information related to the details of the pileup simulation for a given event, filled by MixingModule
Usage: purely descriptive
*/
//
// Original Author:  Mike Hildreth, Notre Dame
//         Created:  April 18, 2011
//
//

#include <vector>
#include <string>
#include <iostream>

class PileupMixingContent {

 public:

  PileupMixingContent(){};

  PileupMixingContent( std::vector<int>& bunchCrossing,
		       std::vector<int>& n_interactions ) 
 {

    bunchCrossing_.reserve(bunchCrossing.size());
    n_interactions_.reserve(bunchCrossing.size());

    for(int inter=0; inter<(int)bunchCrossing.size(); ++inter) {
        bunchCrossing_.push_back(bunchCrossing[inter]);
	n_interactions_.push_back(n_interactions[inter]);
    }
  };



  ~PileupMixingContent(){
    bunchCrossing_.clear();
    n_interactions_.clear();
  };

  const std::vector<int>& getMix_Ninteractions() const { return n_interactions_; }
  const std::vector<int>& getMix_bunchCrossing() const { return bunchCrossing_; }

 private:

  // for "standard" pileup: we have MC Truth information for these


  std::vector<int> bunchCrossing_;
  std::vector<int> n_interactions_;


};

#endif
