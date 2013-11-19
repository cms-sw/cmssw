#ifndef TauAnalysis_MCEmbeddingTools_L1ExtraMixer_h
#define TauAnalysis_MCEmbeddingTools_L1ExtraMixer_h

/** \class L1ExtraMixer
 *
 * Mix L1Extra e/mu/tau/jet and MET objects.
 *
 * NOTE: All L1Extra collections need to be processed by one and the same module,
 *       in order to retain a unique label for all collections.
 *       The mixing if individual collections is implemented via plugins.
 * 
 * \author Christian Veelken, LLR
 *
 * \version $Revision: 1.1 $
 *
 * $Id: L1ExtraMixer.h,v 1.1 2012/10/09 09:00:24 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "TauAnalysis/MCEmbeddingTools/interface/L1ExtraMixerPluginBase.h"

#include <string>

class L1ExtraMixer : public edm::EDProducer 
{
 public:
  explicit L1ExtraMixer(const edm::ParameterSet&);
  ~L1ExtraMixer();

 private:
  virtual void produce(edm::Event&, const edm::EventSetup&);

  std::vector<L1ExtraMixerPluginBase*> plugins_;
};

#endif
