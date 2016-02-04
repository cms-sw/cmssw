#ifndef MU_END_DIGITIZER
#define MU_END_DIGITIZER

// This is CSCDigitizer.h

/** \class CSCDigitizer
 *  Digitizer class for endcap muon CSCs.
 *
 *  \author Rick Wilkinson
 *
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "CLHEP/Random/RandomEngine.h"
#include <boost/utility.hpp>

#include "DataFormats/Common/interface/DetSetVector.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"

class CSCDriftSim;
class CSCWireHitSim;
class CSCStripHitSim;
class CSCWireElectronicsSim;
class CSCStripElectronicsSim;
class CSCLayer;
class CSCNeutronReader;
class CSCStripConditions;

class CSCDigitizer : public boost::noncopyable
{
public:
  typedef edm::DetSetVector<StripDigiSimLink> DigiSimLinks;

  /// configurable parameters
  explicit CSCDigitizer(const edm::ParameterSet & p);
  
  ~CSCDigitizer();

  /**  digitize
   */
  void doAction(MixCollection<PSimHit> & simHits,
                CSCWireDigiCollection & wireDigis,
                CSCStripDigiCollection & stripDigis,
                CSCComparatorDigiCollection & comparators,
                DigiSimLinks & wireDigiSimLinks,
                DigiSimLinks & stripDigiSimLinks);

  /// sets geometry
  void setGeometry(const CSCGeometry * geom) {theCSCGeometry = geom;}

  /// sets the magnetic field
  void setMagneticField(const MagneticField * field);

  void setStripConditions(CSCStripConditions * cond);

  void setParticleDataTable(const ParticleDataTable * pdt);

  void setRandomEngine(CLHEP::HepRandomEngine& engine);

private:
  /// finds the layer in the geometry associated with this det ID
  const CSCLayer * findLayer(int detId) const;

  /// finds which layers, 1-6, aren't in the current list
  std::list<int> layersMissing(const CSCStripDigiCollection & stripDigis) const;

  CSCDriftSim            * theDriftSim;
  CSCWireHitSim          * theWireHitSim;
  CSCStripHitSim         * theStripHitSim;
  CSCWireElectronicsSim  * theWireElectronicsSim;
  CSCStripElectronicsSim * theStripElectronicsSim;
  CSCNeutronReader       * theNeutronReader;
  const CSCGeometry      * theCSCGeometry;
  CSCStripConditions     * theConditions;
  unsigned int theLayersNeeded;
  bool digitizeBadChambers_;
};

#endif

