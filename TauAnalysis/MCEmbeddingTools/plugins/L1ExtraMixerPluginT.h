#ifndef TauAnalysis_MCEmbeddingTools_L1ExtraMixerPluginT_h
#define TauAnalysis_MCEmbeddingTools_L1ExtraMixerPluginT_h

/** \class L1ExtraMixerPluginT
 *
 * Mix L1Extra e/mu/tau/jet objects,
 * i.e. merge the two collections of e/mu/tau/jet objects
 * reconstructed in the original Zmumu event
 * and the embedded event (containing the tau decay products),
 * keeping the 4 highest Pt objects in each collection
 * 
 * \author Christian Veelken, LLR
 *
 * \version $Revision: 1.8 $
 *
 * $Id: L1ExtraMixerT.h,v 1.8 2012/02/13 17:33:04 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "TauAnalysis/MCEmbeddingTools/interface/L1ExtraMixerPluginBase.h"

#include <string>
#include <vector>

template <typename T>
class L1ExtraMixerPluginT : public L1ExtraMixerPluginBase
{
 public:
  explicit L1ExtraMixerPluginT(const edm::ParameterSet&);
  ~L1ExtraMixerPluginT() {}

  virtual void produce(edm::Event&, const edm::EventSetup&);

 private:
  typedef std::vector<T> l1ExtraCollection;
};

#endif
