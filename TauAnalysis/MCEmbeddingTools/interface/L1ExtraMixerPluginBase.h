#ifndef TauAnalysis_MCEmbeddingTools_L1ExtraMixerPluginBase_h
#define TauAnalysis_MCEmbeddingTools_L1ExtraMixerPluginBase_h

/** \class L1ExtraMEtMixerPluginBase
 *
 * Abstract base-class for L1ExtraMixerPlugins
 * 
 * \author Christian Veelken, LLR
 *
 * \version $Revision: 1.8 $
 *
 * $Id: L1ExtraMEtMixer.h,v 1.8 2012/02/13 17:33:04 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <string>

class L1ExtraMixerPluginBase : public edm::EDProducer 
{
 public:
  explicit L1ExtraMixerPluginBase(const edm::ParameterSet&);
  ~L1ExtraMixerPluginBase() {}

  virtual void produce(edm::Event&, const edm::EventSetup&) = 0;

 protected:
  edm::InputTag src1_;
  edm::InputTag src2_;

  std::string instanceLabel_;
};

#include "FWCore/PluginManager/interface/PluginFactory.h"

typedef edmplugin::PluginFactory<L1ExtraMixerPluginBase* (const edm::ParameterSet&)> L1ExtraMixerPluginFactory;

#endif
