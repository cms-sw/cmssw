#ifndef SimMuon_Neutron_SubsystemNeutronReader_h
#define SimMuon_Neutron_SubsystemNeutronReader_h
/**
 \author Rick Wilkinson
  Reads neutron events from a database
 */

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include <vector>

class NeutronReader;

namespace CLHEP {
  class HepRandomEngine;
}

class SubsystemNeutronReader {
public:
  /// the hits will be distributed flat in time between startTime and endTime
  /// eventOccupancy is the expected chamber occupancy from a single
  /// min bias event for each chamber type
  SubsystemNeutronReader(const edm::ParameterSet& pset);
  virtual ~SubsystemNeutronReader();

  /// this class makes sure the same chamberIndex isn't called twice
  /// for an event
  void generateChamberNoise(int chamberType, int chamberIndex, edm::PSimHitContainer& result, CLHEP::HepRandomEngine*);

  void clear() { theChambersDone.clear(); }

protected:
  /// detector-specific way to get the global detector
  /// ID, given the local one.
  virtual int detId(int chamberIndex, int localDetId) = 0;

private:
  NeutronReader* theHitReader;

  /// just makes sure chambers aren't done twice
  std::vector<int> theChambersDone;

  /// in units of 10**34, set by Muon:NeutronLuminosity
  float theLuminosity;
  float theStartTime;
  float theEndTime;
  /// how many collsions happened between theStartTime and theEndTime
  float theEventsInWindow;

  std::vector<double> theEventOccupancy;  // Placed here so ctor init list order OK
};

#endif
