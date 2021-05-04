#include "SimPPS/RPDigiProducer/plugins/RPLinearInduceChargeOnStrips.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

RPLinearInduceChargeOnStrips::RPLinearInduceChargeOnStrips(const edm::ParameterSet &params, RPDetId det_id)
    : det_id_(det_id), theRPDetTopology(params) {
  verbosity_ = params.getParameter<int>("RPVerbosity");
  signalCoupling_.clear();
  double coupling_constant_ = params.getParameter<double>("RPInterStripCoupling");
  signalCoupling_.push_back(coupling_constant_);
  signalCoupling_.push_back((1.0 - coupling_constant_) / 2.);

  no_of_strips_ = theRPDetTopology.DetStripNo();
}

simromanpot::strip_charge_map RPLinearInduceChargeOnStrips::Induce(
    const simromanpot::charge_induced_on_surface &charge_map) {
  theStripChargeMap.clear();
  const double sqrt_2 = sqrt(2.0);
  if (verbosity_)
    edm::LogInfo("RPLinearInduceChargeOnStrips ") << det_id_ << " : Clouds to be induced:" << charge_map.size() << "\n";
  for (simromanpot::charge_induced_on_surface::const_iterator i = charge_map.begin(); i != charge_map.end(); ++i) {
    double hit_pos;
    std::vector<strip_info> relevant_strips =
        theRPDetTopology.GetStripsInvolved((*i).Position().x(), (*i).Position().y(), (*i).Sigma(), hit_pos);
    if (verbosity_) {
      edm::LogInfo("RPLinearInduceChargeOnStrips ")
          << det_id_ << " : relevant_strips" << relevant_strips.size() << "\n";
    }
    for (std::vector<strip_info>::const_iterator j = relevant_strips.begin(); j != relevant_strips.end(); ++j) {
      double strip_begin = (*j).LowerBoarder();
      double strip_end = (*j).HigherBoarder();
      double effic = (*j).EffFactor();
      double sigma = (*i).Sigma();
      unsigned short str_no = (*j).StripNo();

      double charge_on_strip = (TMath::Erfc((strip_begin - hit_pos) / sqrt_2 / sigma) / 2.0 -
                                TMath::Erfc((strip_end - hit_pos) / sqrt_2 / sigma) / 2.0) *
                               (*i).Charge() * effic;
      if (verbosity_)
        edm::LogInfo("RPLinearInduceChargeOnStrips") << "Efficiency " << det_id_ << " :" << effic << "\n";

      for (int k = -signalCoupling_.size() + 1; k < (int)signalCoupling_.size(); ++k) {
        if ((str_no + k) >= 0 && (str_no + k) < no_of_strips_)
          theStripChargeMap[str_no + k] += charge_on_strip * signalCoupling_[abs(k)];
      }
    }
  }

  return theStripChargeMap;
}
