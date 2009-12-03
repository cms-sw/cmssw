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
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "SimMuon/Neutron/src/NeutronWriter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"


class SubsystemNeutronWriter : public edm::EDAnalyzer
{
public:

  explicit SubsystemNeutronWriter(edm::ParameterSet const& pset);

  /// destructor prints statistics on number of events written
  virtual ~SubsystemNeutronWriter();

  void printStats();

  virtual void analyze(edm::Event const& e, edm::EventSetup const& c);

  virtual int localDetId(int globalDetId) const = 0;

  virtual int chamberType(int globalDetId) const = 0;

  virtual int chamberId(int globalDetId) const = 0;

  /// good practice to do once for each chamber type
  void initialize(int chamberType);

protected:


  virtual void writeHits(int chamberType, edm::PSimHitContainer & allSimHits);

  /// helper to add time offsets and local det ID
  void adjust(PSimHit & h, float timeOffset);

  /// updates the counter
  void updateCount(int chamberType);

private:
  NeutronWriter * theHitWriter;
  edm::InputTag theInputTag;
  double theNeutronTimeCut;
  double theTimeWindow;
  int theNEvents;
  bool initialized;
  std::map<int, int> theCountPerChamberType;
};

#endif

