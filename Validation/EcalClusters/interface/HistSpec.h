#ifndef Validation_EcalClusters_HistSpec_h
#define Validation_EcalClusters_HistSpec_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"

struct HistSpec {
  double min;
  double max;
  int bins;

  HistSpec(edm::ParameterSet const& _ps, std::string const& _suffix)
  {
    min = _ps.getParameter<double>("hist_min_" + _suffix);
    max = _ps.getParameter<double>("hist_max_" + _suffix);
    bins = _ps.getParameter<int>("hist_bins_" + _suffix);
  }
};

#endif
