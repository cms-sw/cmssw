#ifndef SimMuon_GEMDigitizer_ME0ReDigiProducer_h
#define SimMuon_GEMDigitizer_ME0ReDigiProducer_h

/*
 * This module smears and discretizes the timing and position of the
 * ME0 pseudo digis.
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/GEMDigi/interface/ME0DigiPreRecoCollection.h"
#include <string>

class ME0Geometry;
class ME0EtaPartition;
class TrapezoidalStripTopology;
class LocalError;
class LocalTag;
template <typename t1, typename t2>
class Point3DBase;
typedef class Point3DBase<float, LocalTag> LocalPoint;

namespace CLHEP {
  class HepRandomEngine;
}

class ME0ReDigiProducer : public edm::stream::EDProducer<> {
private:
  //Class used to define custom geometry, if required
  //assume that all ME0 chambers have the same dimension
  //and for the same layer have the same radial and Z position
  //Good for now, can build in support for more varied geos later
  //if necessary
  class TemporaryGeometry {
  public:
    TemporaryGeometry(const ME0Geometry* geometry,
                      const unsigned int numberOfStrips,
                      const unsigned int numberOfPartitions);
    ~TemporaryGeometry();
    unsigned int findEtaPartition(float locY) const;
    const TrapezoidalStripTopology* getTopo(const unsigned int partIdx) const { return stripTopos[partIdx]; }
    float getPartCenter(const unsigned int partIdx) const;  //position of part. in chamber
    float getCentralTOF(const ME0DetId& me0Id, unsigned int partIdx) const {
      return tofs[me0Id.layer() - 1][partIdx];
    }  //in detId layer numbers stat at 1
    unsigned int numLayers() const { return tofs.size(); }

  private:
    TrapezoidalStripTopology* buildTopo(const std::vector<float>& _p) const;

  private:
    float middleDistanceFromBeam;                       // radiusOfMainPartitionInCenter;
    std::vector<TrapezoidalStripTopology*> stripTopos;  // vector of Topos, one for each part
    std::vector<std::vector<double> > tofs;             //TOF to center of the partition:  [layer][part]
    std::vector<float> partitionTops;                   //Top of each partition in the chamber's local coords
  };

public:
  explicit ME0ReDigiProducer(const edm::ParameterSet& ps);

  ~ME0ReDigiProducer() override;

  void beginRun(const edm::Run&, const edm::EventSetup&) override;

  void produce(edm::Event&, const edm::EventSetup&) override;

  void buildDigis(const ME0DigiPreRecoCollection&,
                  ME0DigiPreRecoCollection&,
                  ME0DigiPreRecoMap&,
                  CLHEP::HepRandomEngine* engine);

private:
  void fillCentralTOFs();
  void getStripProperties(const ME0EtaPartition* etaPart,
                          const ME0DigiPreReco* inDigi,
                          float& tof,
                          int& strip,
                          LocalPoint& digiLocalPoint,
                          LocalError& digiLocalError) const;
  int getCustomStripProperties(const ME0DetId& detId,
                               const ME0DigiPreReco* inDigi,
                               float& tof,
                               int& strip,
                               LocalPoint& digiLocalPoint,
                               LocalError& digiLocalError) const;

  typedef std::tuple<unsigned int, unsigned int, unsigned int> DigiIndicies;
  typedef std::map<DigiIndicies, unsigned int> ChamberDigiMap;
  //fills map...returns -1 if digi is not already in the map
  unsigned int fillDigiMap(
      ChamberDigiMap& chDigiMap, unsigned int bx, unsigned int part, unsigned int strip, unsigned int currentIDX) const;

  //paramters
  const float bxWidth;              // ns
  bool useCusGeoFor1PartGeo;        //Use custom strips and partitions for digitization for single partition geometry
  bool usePads;                     //sets strip granularity to x2 coarser
  unsigned int numberOfStrips;      // Custom number of strips per partition
  unsigned int numberOfPartitions;  // Custom number of partitions per chamber
  double neutronAcceptance;         // fraction of neutron events to keep in event (>= 1 means no filtering)
  double timeResolution;            // smear time by gaussian with this sigma (in ns)....negative for no smearing
  int minBXReadout;                 // Minimum BX to readout
  int maxBXReadout;                 // Maximum BX to readout
  std::vector<int> layerReadout;  // Don't readout layer if entry is 0 (Layer number 1 in the numbering scheme is idx 0)
  bool mergeDigis;                // Keep only one digi at the same chamber, strip, partition, and BX
  edm::EDGetTokenT<ME0DigiPreRecoCollection> token;

  bool useBuiltinGeo;
  const ME0Geometry* geometry;
  TemporaryGeometry* tempGeo;
  std::vector<std::vector<double> > tofs;  //used for built in geo
};

#endif
