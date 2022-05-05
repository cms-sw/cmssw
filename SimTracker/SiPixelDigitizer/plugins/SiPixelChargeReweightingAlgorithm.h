#ifndef SimTracker_SiPixelDigitizer_SiPixelChargeReweightingAlgorithm_h
#define SimTracker_SiPixelDigitizer_SiPixelChargeReweightingAlgorithm_h

#include "CondFormats/DataRecord/interface/SiPixel2DTemplateDBObjectRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixel2DTemplateDBObject.h"
#include "FWCore/Framework/interface/FrameworkfwdMostUsed.h"
#include "SimTracker/SiPixelDigitizer/plugins/SiPixelDigitizerAlgorithm.h"

// forward declarations
class DetId;
class GaussianTailNoiseGenerator;
class PixelDigi;
class PixelDigiSimLink;
class PixelGeomDetUnit;
class SiG4UniversalFluctuation;

class SiPixelChargeReweightingAlgorithm {
public:
  SiPixelChargeReweightingAlgorithm(const edm::ParameterSet& conf, edm::ConsumesCollector iC);
  ~SiPixelChargeReweightingAlgorithm();

  // initialization that cannot be done in the constructor
  void init(const edm::EventSetup& es);

  typedef std::map<int, SiPixelDigitizerAlgorithm::Amplitude, std::less<int> > signal_map_type;  // from Digi.Skel.
  typedef signal_map_type::iterator signal_map_iterator;                                         // from Digi.Skel.
  typedef signal_map_type::const_iterator signal_map_const_iterator;                             // from Digi.Skel.

  bool hitSignalReweight(const PSimHit& hit,
                         std::map<int, float, std::less<int> >& hit_signal,
                         const size_t hitIndex,
                         const size_t hitIndex4CR,
                         const unsigned int tofBin,
                         const PixelTopology* topol,
                         uint32_t detID,
                         signal_map_type& theSignal,
                         unsigned short int processType,
                         const bool& boolmakeDigiSimLinks);

  bool lateSignalReweight(const PixelGeomDetUnit* pixdet,
                          std::vector<PixelDigi>& digis,
                          PixelSimHitExtraInfo& loopTempSH,
                          signal_map_type& theNewDigiSignal,
                          const TrackerTopology* tTopo,
                          CLHEP::HepRandomEngine* engine);

private:
  // Internal typedef
  typedef std::map<uint32_t, signal_map_type> signalMaps;
  typedef GloballyPositioned<double> Frame;
  typedef std::vector<edm::ParameterSet> Parameters;
  typedef boost::multi_array<float, 2> array_2d;

  // Variables and objects for the charge reweighting using 2D templates
  SiPixelTemplate2D templ2D;
  std::vector<bool> xdouble;
  std::vector<bool> ydouble;
  std::vector<float> track;
  int IDnum, IDden;

  const bool UseReweighting;
  bool applyLateReweighting_;
  const bool PrintClusters;
  const bool PrintTemplates;

  static constexpr float cmToMicrons = 10000.f;

  std::vector<SiPixelTemplateStore2D> templateStores_;

  edm::ESGetToken<SiPixel2DTemplateDBObject, SiPixel2DTemplateDBObjectRcd> SiPixel2DTemp_den_token_;
  edm::ESGetToken<SiPixel2DTemplateDBObject, SiPixel2DTemplateDBObjectRcd> SiPixel2DTemp_num_token_;
  const SiPixel2DTemplateDBObject* dbobject_den;
  const SiPixel2DTemplateDBObject* dbobject_num;

  // methods for charge reweighting in irradiated sensors
  int PixelTempRewgt2D(int id_gen, int id_rewgt, array_2d& cluster);
  void printCluster(array_2d& cluster);
  void printCluster(float arr[BXM2][BYM2]);
  void printCluster(float arr[TXSIZE][TYSIZE]);
};

#endif
