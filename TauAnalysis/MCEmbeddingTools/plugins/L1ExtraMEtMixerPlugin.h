#ifndef TauAnalysis_MCEmbeddingTools_L1ExtraMEtMixerPlugin_h
#define TauAnalysis_MCEmbeddingTools_L1ExtraMEtMixerPlugin_h

/** \class L1ExtraMEtMixerPlugin
 *
 * Mix L1Extra MET objects,
 * i.e. add MET reconstructed in original Zmumu event
 * and negative vectorial momentum sum of embedded tau decay products
 * 
 * \author Christian Veelken, LLR
 *
 * \version $Revision: 1.1 $
 *
 * $Id: L1ExtraMEtMixerPlugin.h,v 1.1 2012/10/09 09:00:24 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "TauAnalysis/MCEmbeddingTools/interface/L1ExtraMixerPluginBase.h"

#include <string>

class L1ExtraMEtMixerPlugin : public L1ExtraMixerPluginBase
{
 public:
  explicit L1ExtraMEtMixerPlugin(const edm::ParameterSet&);
  ~L1ExtraMEtMixerPlugin() {}

  virtual void registerProducts(edm::EDProducer&);

  virtual void produce(edm::Event&, const edm::EventSetup&);
};

#endif
