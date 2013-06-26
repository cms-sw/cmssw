#ifndef RHSTOPDUMP_H
#define RHSTOPDUMP_H 1
//
// Dump stopping points from the Event into ASCII file
// F.Ratnikov, Apr. 8, 2010
// $Id: RHStopDump.h,v 1.1 2010/04/09 11:38:36 fedor Exp $
//

#include <fstream>

#include "FWCore/Framework/interface/EDAnalyzer.h"

class RHStopDump : public edm::EDAnalyzer {
 public:
  explicit RHStopDump(const edm::ParameterSet&);
  virtual ~RHStopDump() {};
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
 private:
  std::ofstream mStream;
  std::string mProducer;
};

#endif
