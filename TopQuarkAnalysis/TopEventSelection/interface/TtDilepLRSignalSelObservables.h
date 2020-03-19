#ifndef TtDilepLRSignalSelObservables_h
#define TtDilepLRSignalSelObservables_h

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>
#include <string>
#include <vector>

#include "AnalysisDataFormats/TopObjects/interface/TtDilepEvtSolution.h"

class TtDilepLRSignalSelObservables {
public:
  TtDilepLRSignalSelObservables(edm::ConsumesCollector&& iC,
                                const edm::EDGetTokenT<std::vector<pat::Jet> >& jetSourceToken);
  ~TtDilepLRSignalSelObservables();

  typedef std::pair<unsigned int, bool> IntBoolPair;
  std::vector<IntBoolPair> operator()(TtDilepEvtSolution&, const edm::Event& iEvent, bool matchOnly = false);

private:
  typedef std::pair<unsigned int, double> IntDblPair;

  double delta(double phi1, double phi2);
  void fillMinMax(double v1,
                  double v2,
                  int obsNbr,
                  std::vector<IntDblPair>& varList,
                  bool match1,
                  bool match2,
                  std::vector<IntBoolPair>& matchList);

  edm::EDGetTokenT<std::vector<pat::Jet> > jetSourceToken_;
  edm::EDGetTokenT<TtGenEvent> genEvtToken_;

  std::vector<IntDblPair> evtselectVarVal;
  std::vector<IntBoolPair> evtselectVarMatch;
  int count1, count2, count3, count4, count5;
};

#endif
