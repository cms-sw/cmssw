#ifndef SubsystemNeutronWriter_h
#define SubsystemNeutronWriter_h

/** theSubsystemNeutronWriter stores "events"
 * which consist of a list of SimHits,
 * grouped by detector type.  These can then
 * be read back to model neutron background
 * int muon chambers.
 *
 *  You can specify the cut on how long after the
 * signal event to define something as a Neutron Event
 * with the configurable Muon:NeutronTimeCut
 */

#include <vector>
#include <map>
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace CLHEP {
  class HepRandomEngine;
}

class NeutronWriter;

/// doesn't have to be a producer.  Can act as an analyzer, too.
class SubsystemNeutronWriter : public edm::stream::EDProducer<> {
public:
  explicit SubsystemNeutronWriter(edm::ParameterSet const& pset);

  /// destructor prints statistics on number of events written
  ~SubsystemNeutronWriter() override;

  void printStats();

  void produce(edm::Event& e, edm::EventSetup const& c) override;

  virtual int localDetId(int globalDetId) const = 0;

  virtual int chamberType(int globalDetId) const = 0;

  virtual int chamberId(int globalDetId) const = 0;

  /// decides whether this cluster is good enough to be included
  virtual bool accept(const edm::PSimHitContainer& cluster) const = 0;

  /// good practice to do once for each chamber type
  void initialize(int chamberType);

protected:
  virtual void writeHits(int chamberType, edm::PSimHitContainer& chamberHits, CLHEP::HepRandomEngine*);

  void writeCluster(int chamberType, const edm::PSimHitContainer& cluster);

  /// helper to add time offsets and local det ID
  void adjust(PSimHit& h, float timeOffset, float smearing);

  /// updates the counter
  void updateCount(int chamberType);

private:
  NeutronWriter* theHitWriter;
  bool useRandFlat;
  const edm::InputTag theInputTag;
  const double theNeutronTimeCut;
  const double theTimeWindow;
  const double theT0;
  const edm::EDGetTokenT<edm::PSimHitContainer> hitToken_;
  int theNEvents;
  bool initialized;
  // true means to translate DetId into just layer number, e.g., 1-6 in CSC
  bool useLocalDetId_;
  std::map<int, int> theCountPerChamberType;
};

#endif
