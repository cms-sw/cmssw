#include "TauAnalysis/MCEmbeddingTools/interface/L1ExtraMixerPluginBase.h"

L1ExtraMixerPluginBase::L1ExtraMixerPluginBase(const edm::ParameterSet& cfg)
{
  edm::InputTag src1 = cfg.getParameter<edm::InputTag>("src1");
  edm::InputTag src2 = cfg.getParameter<edm::InputTag>("src2");

  instanceLabel_ = cfg.getParameter<std::string>("instanceLabel");

  // CV: update instance labels
  //    (this logic guarantees that instance labels of the two input collections 
  //     and the output collection always matches in all derived classes)
  src1_ = edm::InputTag(src1.label(), instanceLabel_, src1.process());
  src2_ = edm::InputTag(src2.label(), instanceLabel_, src2.process());
}

#include "FWCore/Framework/interface/MakerMacros.h"

EDM_REGISTER_PLUGINFACTORY(L1ExtraMixerPluginFactory, "L1ExtraMixerPluginFactory");

