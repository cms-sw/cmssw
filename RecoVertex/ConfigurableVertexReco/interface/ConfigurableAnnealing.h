#ifndef ConfigurableAnnealing_H
#define ConfigurableAnnealing_H

#include "RecoVertex/VertexTools/interface/AnnealingSchedule.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <vector>

class ConfigurableAnnealing : public AnnealingSchedule {
  /**
   *  An annealing schedule that is completely configurable
   *  via edm::ParameterSet
   */

public:
  ConfigurableAnnealing(const edm::ParameterSet&);
  ~ConfigurableAnnealing() override;
  ConfigurableAnnealing(const ConfigurableAnnealing&);
  void anneal() override;
  void resetAnnealing() override;
  double phi(double chi2) const override;
  double weight(double chi2) const override;
  double cutoff() const override;
  double currentTemp() const override;
  double initialTemp() const override;
  bool isAnnealed() const override;
  void debug() const override;

  ConfigurableAnnealing* clone() const override;

private:
  AnnealingSchedule* theImpl;
};

#endif
