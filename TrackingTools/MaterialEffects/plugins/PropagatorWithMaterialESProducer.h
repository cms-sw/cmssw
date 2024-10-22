#ifndef TrackingTools_ESProducers_PropagatorWithMaterialESProducer_h
#define TrackingTools_ESProducers_PropagatorWithMaterialESProducer_h

/** \class PropagatorWithMaterialESProducer
 *  ESProducer for PropagatorWithMaterial.
 *
 *  \author cerati
 */

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include <memory>

class PropagatorWithMaterialESProducer : public edm::ESProducer {
public:
  PropagatorWithMaterialESProducer(const edm::ParameterSet &p);

  std::unique_ptr<Propagator> produce(const TrackingComponentsRecord &);

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> mfToken_;
  const double mass_;
  const double maxDPhi_;
  const double ptMin_;
  const PropagationDirection dir_;
  const bool useRK_;
  const bool useOldAnalPropLogic_;
};

#endif
