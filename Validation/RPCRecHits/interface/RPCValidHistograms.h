#ifndef Validation_RPCRecHits_RPCValidHistograms_H
#define Validation_RPCRecHits_RPCValidHistograms_H

#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <string>

struct RPCValidHistograms
{
  typedef MonitorElement* MEP;

  RPCValidHistograms()
  {
    booked_ = false;
  };

  void bookHistograms(DQMStore* dbe, const std::string subDir);

  MEP clusterSize;

  // Number of hits
  MEP nRefHit_W, nRefHit_D;
  MEP nRecHit_W, nRecHit_D;

  MEP nRefHit_WvsR, nRefHit_DvsR;
  MEP nRecHit_WvsR, nRecHit_DvsR;

  MEP nMatchedRefHit_W, nMatchedRefHit_D;
//  MEP nMatchedRecHit_W, nMatchedRecHit_D;

  MEP nMatchedRefHit_WvsR;//, nMatchedRecHit_WvsR;
  MEP nMatchedRefHit_DvsR;//, nMatchedRecHit_DvsR;

  MEP nUnMatchedRefHit_W, nUnMatchedRefHit_D;
  MEP nUnMatchedRecHit_W, nUnMatchedRecHit_D;

  MEP nUnMatchedRefHit_WvsR, nUnMatchedRecHit_WvsR;
  MEP nUnMatchedRefHit_DvsR, nUnMatchedRecHit_DvsR;

  // Residuals
  MEP res_W, res_D;
  MEP res2_W, res2_D;
  MEP res2_WR, res2_DR;

  // Pulls
  MEP pull_W, pull_D;
  MEP pull2_W, pull2_D;
  MEP pull2_WR, pull2_DR;

private:
  bool booked_;
};

#endif

