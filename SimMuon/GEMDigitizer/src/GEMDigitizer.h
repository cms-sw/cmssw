#ifndef SimMuon_GEMDigitizer_h
#define SimMuon_GEMDigitizer_h

/** \class GEMDigitizer
 *  Digitizer for GEM
 *
 *  \author Vadim Khotilovich
 *
 */

#include "DataFormats/Common/interface/DetSetVector.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include <string>
#include "CLHEP/Random/RandomEngine.h"

namespace edm{
  class ParameterSet;
}

class GEMEtaPartition;
class GEMSim;
class GEMSimSetUp;
class GEMGeometry;

class GEMDigitizer
{
public:
  
  typedef edm::DetSetVector<StripDigiSimLink> StripDigiSimLinks;

  GEMDigitizer(const edm::ParameterSet& config, CLHEP::HepRandomEngine&);

  ~GEMDigitizer();

  void digitize(MixCollection<PSimHit> & simHits,
                GEMDigiCollection & rpcDigis,
                StripDigiSimLinks & digiSimLinks);

  /// sets geometry
  void setGeometry(const GEMGeometry *geom) {geometry_ = geom;}

  void setGEMSimSetUp(GEMSimSetUp *simsetup) {simSetUp_ = simsetup;}

  GEMSimSetUp* getGEMSimSetUp() { return simSetUp_; }
  
  /// finds the GEM det unit in the geometry associated with this det ID
  const GEMEtaPartition * findDet(int detId) const;

private:

  const GEMGeometry * geometry_;
  GEMSim* gemSim_;
  GEMSimSetUp * simSetUp_;
  std::string modelName_;
};

#endif
