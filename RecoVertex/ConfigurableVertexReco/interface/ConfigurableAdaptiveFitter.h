#ifndef _ConfigurableAdaptiveFitter_H_
#define _ConfigurableAdaptiveFitter_H_

#include "RecoVertex/ConfigurableVertexReco/interface/AbstractConfFitter.h"

/**
 *  Wrap any VertexFitter into the VertexReconstructor interface
 */

class ConfigurableAdaptiveFitter : public AbstractConfFitter {
public:
  /**
     *  Values that are respected:
     *  sigmacut: The sqrt(chi2_cut) criterion. Default: 3.0
     *  ratio:   The annealing ratio. Default: 0.25
     *  Tini:    The initial temparature. Default: 256
     */
  ConfigurableAdaptiveFitter();
  void configure(const edm::ParameterSet&) override;
  ConfigurableAdaptiveFitter(const ConfigurableAdaptiveFitter& o);
  ~ConfigurableAdaptiveFitter() override;
  ConfigurableAdaptiveFitter* clone() const override;
  edm::ParameterSet defaults() const override;
};

#endif
