#ifndef PileupVertexContent_h
#define PileupVertexContent_h
// -*- C++ -*-
//
// Package:     PileupVertexContent
// Class  :     PileupVertexContent
//
/**\class PileupVertexContent PileupVertexContent.h SimDataFormats/PileupVertexContent/interface/PileupVertexContent.h

Description: contains information related to the details of the pileup simulation for a given event, filled by Special "Digitizer" that has access to each pileup event
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
#include "DataFormats/Math/interface/LorentzVector.h"

class PileupVertexContent {
public:
  PileupVertexContent(){};

  PileupVertexContent(const std::vector<float>& pT_hat,
                      const std::vector<float>& z_Vtx,
                      const std::vector<float>& t_Vtx)
      : pT_hats_(pT_hat), z_Vtxs_(z_Vtx), t_Vtxs_(t_Vtx){};

  ~PileupVertexContent() {
    pT_hats_.clear();
    z_Vtxs_.clear();
    t_Vtxs_.clear();
  };

  const std::vector<float>& getMix_pT_hats() const { return pT_hats_; }
  const std::vector<float>& getMix_z_Vtxs() const { return z_Vtxs_; }
  const std::vector<float>& getMix_t_Vtxs() const { return t_Vtxs_; }

private:
  // for "standard" pileup: we have MC Truth information for these

  std::vector<float> pT_hats_;
  std::vector<float> z_Vtxs_;
  std::vector<float> t_Vtxs_;  // may be empty if time information is not available
};

#endif
