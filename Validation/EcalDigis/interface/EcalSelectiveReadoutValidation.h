#ifndef EcalSelectiveReadoutValidation_H
#define EcalSelectiveReadoutValidation_H

/*
 * \file EcalSelectiveReadoutValidation.h
 *
 *
 */

#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/EcalDetId/interface/EcalScDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"

#include "Validation/EcalDigis/src/CollHandle.h"

#include <string>
#include <set>
#include <fstream>

class EBDetId;
class EEDetId;
class EcalElectronicsMapping;
class EcalTrigTowerConstituentsMap;

class EcalSelectiveReadoutValidation : public DQMOneEDAnalyzer<> {
  typedef EcalRecHitCollection RecHitCollection;
  typedef EcalRecHit RecHit;

public:
  /// Constructor
  EcalSelectiveReadoutValidation(const edm::ParameterSet& ps);

  /// Destructor
  ~EcalSelectiveReadoutValidation() override;
  void dqmBeginRun(edm::Run const&, edm::EventSetup const&) override;
  void bookHistograms(DQMStore::IBooker& i, edm::Run const&, edm::EventSetup const&) override;

protected:
  /// Analyzes the event.
  void analyze(edm::Event const& e, edm::EventSetup const& c) override;

  void dqmEndRun(const edm::Run& r, const edm::EventSetup& c) override;

private:
  ///distinguishes barral and endcap of ECAL.
  enum subdet_t { EB, EE };

  /** Accumulates statitics for data volume analysis. To be called for each
   * ECAL digi. See anaDigiInit().
   */
  template <class T, class U>
  void anaDigi(const T& frame, const U& srFlagColl);

  /** Initializes statistics accumalator for data volume analysis. To
   * be call at start of each event analysis.
   */
  void anaDigiInit();

  /** Data volume analysis. To be called for each event.
   * @param event EDM event
   * @param es event setup
   */
  void analyzeDataVolume(const edm::Event& e, const edm::EventSetup& es);

  /** ECAL barrel data analysis. To be called for each event.
   * @param event EDM event
   * @param es event setup
   */
  void analyzeEB(const edm::Event& event, const edm::EventSetup& es);

  /** ECAL endcap data analysis. To be called for each event.
   * @param event EDM event
   * @param es event setup
   */
  void analyzeEE(const edm::Event& event, const edm::EventSetup& es);

  /** Trigger primitive analysis. To be called for each event.
   * @param event EDM event
   * @param es event setup
   */
  void analyzeTP(const edm::Event& event, const edm::EventSetup& es);

  //  /** Selective Readout decisions Validation
  //    * @param event EDM event
  //    * @param es event setup
  //    */
  //   void SRFlagValidation(const edm::Event& event, const edm::EventSetup& es);

  /** Energy reconstruction from ADC samples.
   * @param frame the ADC sample of an ECA channel
   */
  double frame2Energy(const EcalDataFrame& frame) const;

  /** Energy reconstruction from ADC samples to be used for trigger primitive
   * estimate.
   * @param frame the ADC sample of an ECA channel
   * @param offset time offset. To be used to evaluate energy of the event
   * previous (offset=-1) and next (offset=+1) to the triggered one.
   */
  template <class T>
  double frame2EnergyForTp(const T& frame, int offset = 0) const;

  //   double getEcalEventSize(double nReadXtals) const{
  //     return getDccOverhead(EB)*nEbDccs+getDccOverhead(EE)*nEeDccs
  //       + nReadXtals*getBytesPerCrystal()
  //       + (nEeRus+nEbRus)*8;
  //   }

  /** Computes the size of an ECAL barrel event fragment.
   * @param nReadXtals number of read crystal channels
   * @return the event fragment size in bytes
   */
  double getEbEventSize(double nReadXtals) const;

  /** Computes the size of an ECAL endcap event fragment.
   * @param nReadXtals number of read crystal channels
   * @return the event fragment size in bytes
   */
  double getEeEventSize(double nReadXtals) const;

  /** Gets the size in bytes fixed-size part of a DCC event fragment.
   * @return the fixed size in bytes.
   */
  double getDccOverhead(subdet_t subdet) const {
    //  return (subdet==EB?34:25)*8;
    return (subdet == EB ? 34 : 52) * 8;
  }

  /** Gets the number of bytes per crystal channel of the event part
   * depending on the number of read crystal channels.
   * @return the number of bytes.
   */
  double getBytesPerCrystal() const { return 3 * 8; }

  /** Gets the size of an DCC event fragment.
   * @param iDcc0 the DCC logical number starting from 0.
   * @param nReadXtals number of read crystal channels.
   * @return the DCC event fragment size in bytes.
   */
  double getDccEventSize(int iDcc0, double nReadXtals) const {
    subdet_t subdet;
    if (iDcc0 < 9 || iDcc0 >= 45) {
      subdet = EE;
    } else {
      subdet = EB;
    }
    //     return getDccOverhead(subdet)+nReadXtals*getBytesPerCrystal()
    //       + getRuCount(iDcc0)*8;
    return getDccOverhead(subdet) + getDccSrDependentPayload(iDcc0, getRuCount(iDcc0), nReadXtals);
  }

  /** Gets DCC event fragment payload depending on the channel selection
   * made by the selective readout.
   * @param iDcc0 the DCC logical number starting from 0.
   * @param nReadRus number of read-out RUs
   * @param nReadXtals number of read-out crystal channels.
   * @return the DCC event fragment payload in bytes.
   */
  double getDccSrDependentPayload(int iDcc0, double nReadRus, double nReadXtals) const {
    return nReadXtals * getBytesPerCrystal() + nReadRus * 8;
  }

  /** Gets the number of readout unit read by a DCC. A readout unit
   * correspond to an active DCC input channel.
   * @param iDcc0 DCC logical number starting from 0.
   */
  int getRuCount(int iDcc0) const;

  /** Reads the data collections from the event. Called at start
   * of each event analysis.
   * @param event the EDM event.
   */
  void readAllCollections(const edm::Event& e);

  /** Computes trigger primitive estimates. A sum of crystal deposited
   * transverse energy is performed.
   * @param es event setup
   * @param ebDigis the ECAL barrel unsuppressed digi to use for the
   * computation
   * @param ebDigis the ECAL endcap unsuppressed digi to use for the
   * computation
   */
  void setTtEtSums(const edm::EventSetup& es, const EBDigiCollection& ebDigis, const EEDigiCollection& eeDigis);

  //   /** Retrieves the logical number of the DCC reading a given crystal channel.
  //    * @param xtarId crystal channel identifier
  //    * @return the DCC logical number starting from 1.
  //    */
  //   unsigned dccNum(const DetId& xtalId) const;

  /** Retrieves the DCC channel reading out a crystal, the
   * crystals of a barrel trigger tower or the crystals,
   * of an endcap supercrystal.
   * @param xtarId crystal channel, barrel trigger tower or
   * endcap supercrystal identifier
   * @return pair of (DCC ID, DCC channel)
   */
  std::pair<int, int> dccCh(const DetId& xtalId) const;

  /** Converts a std CMSSW crystal eta index to a c-array index (starting from
   * zero and without hole).
   */
  int iEta2cIndex(int iEta) const { return (iEta < 0) ? iEta + 85 : iEta + 84; }

  /** Converts a std CMSSW crystal phi index to a c-array index (starting from
   * zero and without hole).
   */
  int iPhi2cIndex(int iPhi) const {
    //    return iPhi-1;
    int iPhi0 = iPhi - 11;
    if (iPhi0 < 0)
      iPhi0 += 360;
    return iPhi0;
  }

  /** Converts a std CMSSW crystal x or y index to a c-array index (starting
   * from zero and without hole).
   */
  int iXY2cIndex(int iX) const { return iX - 1; }

  /** converse of iXY2cIndex() method.
   */
  int cIndex2iXY(int iX0) const { return iX0 + 1; }

  /** converse of iEta2cIndex() method.
   */
  int cIndex2iEta(int i) const { return (i < 85) ? i - 85 : i - 84; }

  /** converse of iPhi2cIndex() method.
   */
  int cIndex2iPhi(int i) const { return (i + 11) % 360; }

  /**Transforms CMSSW eta ECAL TT indices to indices starting at 0
   * to use for c-array or vector.
   * @param iEta CMSSW eta index (numbering -28...-1,28...56)
   * @return index in numbering from 0 to 55
   */
  int iTtEta2cIndex(int iEta) const { return (iEta < 0) ? iEta + 28 : iEta + 27; }

  /**Transforms CMSSW phi ECAL crystal indices to indices starting at 0
   * to use for c-array or vector.
   * @param iPhi CMSSW phi index (numbering 1...72)
   * @return index in numbering 0...71
   */
  int iTtPhi2cIndex(int iPhi) const {
    return iPhi - 1;
    //int iPhi0 = iPhi - 3;
    //if(iPhi0<0) iPhi0 += 72;
    //return iPhi0;
  }

  /** converse of iTtEta2cIndex() method.
   */
  int cIndex2iTtEta(int i) const { return (i < 27) ? i - 28 : i - 27; }

  /** converse of iTtPhi2cIndex() method.
   */
  int cIndex2iTtPhi(int i) const { return i + 1; }

  //@{
  /** Retrives the readout unit, a trigger tower in the barrel case,
   * and a supercrystal in the endcap case, a given crystal belongs to.
   * @param xtalId identifier of the crystal
   * @return identifer of the supercrystal or of the trigger tower.
   */
  EcalTrigTowerDetId readOutUnitOf(const EBDetId& xtalId) const;

  EcalScDetId readOutUnitOf(const EEDetId& xtalId) const;
  //@}

  /** Emulates the DCC zero suppression FIR filter. If one of the time sample
   * is not in gain 12, numeric_limits<int>::max() is returned.
   * @param frame data frame
   * @param firWeights TAP weights
   * @param firstFIRSample index (starting from 1) of the first time
   * sample to be used in the filter
   * @param saturated if not null, *saturated is set to true if all the time
   * sample are not in gain 12 and set to false otherwise.
   * @return FIR output or numeric_limits<int>::max().
   */
  static int dccZsFIR(const EcalDataFrame& frame,
                      const std::vector<int>& firWeights,
                      int firstFIRSample,
                      bool* saturated = nullptr);

  /** Computes the ZS FIR filter weights from the normalized weights.
   * @param normalizedWeights the normalized weights
   * @return the computed ZS filter weights.
   */
  static std::vector<int> getFIRWeights(const std::vector<double>& normalizedWeights);

  //@{
  /** Wrappers to the book methods of the DQMStore DQM
   *  histogramming interface.
   */
  MonitorElement* bookFloat(DQMStore::IBooker&, const std::string& name);

  MonitorElement* book1D(
      DQMStore::IBooker&, const std::string& name, const std::string& title, int nbins, double xmin, double xmax);

  MonitorElement* book2D(DQMStore::IBooker&,
                         const std::string& name,
                         const std::string& title,
                         int nxbins,
                         double xmin,
                         double xmax,
                         int nybins,
                         double ymin,
                         double ymax);

  MonitorElement* bookProfile(
      DQMStore::IBooker&, const std::string& name, const std::string& title, int nbins, double xmin, double xmax);

  MonitorElement* bookProfile2D(DQMStore::IBooker&,
                                const std::string& name,
                                const std::string& title,
                                int nbinx,
                                double xmin,
                                double xmax,
                                int nbiny,
                                double ymin,
                                double ymax,
                                const char* option = "");
  //@}

  //@{
  /** Wrapper to fill methods of DQM monitor elements.
   */
  void fill(MonitorElement* me, float x) {
    if (me)
      me->Fill(x);
  }
  void fill(MonitorElement* me, float x, float yw) {
    if (me)
      me->Fill(x, yw);
  }
  void fill(MonitorElement* me, float x, float y, float zw) {
    if (me)
      me->Fill(x, y, zw);
  }
  void fill(MonitorElement* me, float x, float y, float z, float w) {
    if (me)
      me->Fill(x, y, z, w);
  }
  //@}

  void initAsciiFile();

  /** Updates estimate of L1A rate
   * @param event EDM event
   */
  void updateL1aRate(const edm::Event& event);

  /** Gets L1A rate estimate.
   * @see updateL1aRate(const edm::Event&)
   * @return L1A rate estimate
   */
  double getL1aRate() const;

private:
  /** Used to store barrel crystal channel information
   */
  struct energiesEb_t {
    double simE;      ///sim hit energy sum
    double noZsRecE;  ///energy reconstructed from unsuppressed digi
    double recE;      ///energy reconstructed from zero-suppressed digi
    //    EBDigiCollection::const_iterator itNoZsFrame; //
    int simHit;   ///number of sim hits
    double phi;   ///phi crystal position in degrees
    double eta;   ///eta crystal position
    bool gain12;  //all MGPA samples at gain 12?
  };

  /** Used to store endcap crystal channel information
   */
  struct energiesEe_t {
    double simE;      ///sim hit energy sum
    double noZsRecE;  ///energy reconstructed from unsuppressed digi
    double recE;      ///energy reconstructed from zero-suppressed digi
    //    EEDigiCollection::const_iterator itNoZsFrame;
    int simHit;   ///number of sim hits
    double phi;   ///phi crystal position in degrees
    double eta;   ///eta crystal position
    bool gain12;  //all MGPA samples at gain 12?
  };

  /// number of bytes in 1 kByte:
  static const int kByte_ = 1024;

  ///Total number of DCCs
  static const unsigned nDccs_ = 54;

  ///Number of input channels of a DCC
  // = maximum number of RUs read by a DCC
  static const unsigned nDccChs_ = 68;

  //Lower bound of DCC ID range
  static const int minDccId_ = 1;

  //Upper bound of DCC ID range
  static const int maxDccId_ = minDccId_ + nDccs_ - 1;

  /// number of DCCs for EB
  static const int nEbDccs = 36;

  /// number of DCCs for EE
  static const int nEeDccs = 18;

  ///number of RUs for EB
  static const int nEbRus = 36 * 68;

  ///number of RUs for EE
  static const int nEeRus = 2 * (34 + 32 + 33 + 33 + 32 + 34 + 33 + 34 + 33);

  ///number of RUs for each DCC
  static const int nDccRus_[nDccs_];

  ///number of endcaps
  static const int nEndcaps = 2;

  ///number of crystals along Eta in EB
  static const int nEbEta = 170;

  ///number of crystals along Phi in EB
  static const int nEbPhi = 360;

  ///EE crystal grid size along X
  static const int nEeX = 100;

  ///EE crystal grid size along Y
  static const int nEeY = 100;

  ///Number of crystals along an EB TT
  static const int ebTtEdge = 5;

  ///Number of crystals along a supercrystal edge
  static const int scEdge = 5;

  ///Number of Trigger Towers in an endcap along Eta
  static const int nOneEeTtEta = 11;

  ///Number of Trigger Towers in barrel along Eta
  static const int nEbTtEta = 34;

  ///Number of Trigger Towers along Eta
  static const int nTtEta = 2 * nOneEeTtEta + nEbTtEta;

  ///Number of Trigger Towers along Phi
  static const int nTtPhi = 72;

  ///Number of crystals per Readout Unit excepted partial SCs
  static const int nMaxXtalPerRu = 25;
  ///Conversion factor from radian to degree
  static const double rad2deg;

  ///Verbosity switch
  bool verbose_;

  ///Output file for histograms
  std::string outputFile_;

  ///Switch for collection-not-found warning
  bool collNotFoundWarn_;

  ///Output ascii file name for unconsistency between SRFs read from data
  ///and SRF obtained by rerunning SRP algorithm on TTFs.
  std::string srpAlgoErrorLogFileName_;

  ///Output ascii file name for unconsistency between SRFs and actual number
  ///of read-out crystals.
  std::string srApplicationErrorLogFileName_;

  ///Output ascii file for unconsistency on SR flags
  std::ofstream srpAlgoErrorLog_;

  ///Output ascii file for unconsistency between Xtals and RU Flags
  std::ofstream srApplicationErrorLog_;

  ///File to log ZS and other errors
  std::ofstream zsErrorLog_;

  //@{
  /** The event product collections.
   */
  CollHandle<EBDigiCollection> ebDigis_;
  CollHandle<EEDigiCollection> eeDigis_;
  CollHandle<EBDigiCollection> ebNoZsDigis_;
  CollHandle<EEDigiCollection> eeNoZsDigis_;
  CollHandle<EBSrFlagCollection> ebSrFlags_;
  CollHandle<EESrFlagCollection> eeSrFlags_;
  CollHandle<EBSrFlagCollection> ebComputedSrFlags_;
  CollHandle<EESrFlagCollection> eeComputedSrFlags_;
  CollHandle<std::vector<PCaloHit> > ebSimHits_;
  CollHandle<std::vector<PCaloHit> > eeSimHits_;
  CollHandle<EcalTrigPrimDigiCollection> tps_;
  CollHandle<RecHitCollection> ebRecHits_;
  CollHandle<RecHitCollection> eeRecHits_;
  CollHandle<FEDRawDataCollection> fedRaw_;
  //@}

  //@{
  /** For L1A rate estimate
   */
  int64_t tmax;
  int64_t tmin;
  int64_t l1aOfTmin;
  int64_t l1aOfTmax;
  bool l1aRateErr;
  //@}

  //@{
  /** The histograms
   */
  MonitorElement* meDccVol_;
  MonitorElement* meDccLiVol_;
  MonitorElement* meDccHiVol_;
  MonitorElement* meDccVolFromData_;
  MonitorElement* meVol_;
  MonitorElement* meVolB_;
  MonitorElement* meVolE_;
  MonitorElement* meVolBLI_;
  MonitorElement* meVolELI_;
  MonitorElement* meVolLI_;
  MonitorElement* meVolBHI_;
  MonitorElement* meVolEHI_;
  MonitorElement* meVolHI_;
  MonitorElement* meChOcc_;

  MonitorElement* meTp_;
  MonitorElement* meTtf_;
  MonitorElement* meTtfVsTp_;
  MonitorElement* meTtfVsEtSum_;
  MonitorElement* meTpVsEtSum_;

  MonitorElement* meEbRecE_;
  MonitorElement* meEbEMean_;
  MonitorElement* meEbNoise_;
  MonitorElement* meEbSimE_;
  MonitorElement* meEbRecEHitXtal_;
  MonitorElement* meEbRecVsSimE_;
  MonitorElement* meEbNoZsRecVsSimE_;

  MonitorElement* meEeRecE_;
  MonitorElement* meEeEMean_;
  MonitorElement* meEeNoise_;
  MonitorElement* meEeSimE_;
  MonitorElement* meEeRecEHitXtal_;
  MonitorElement* meEeRecVsSimE_;
  MonitorElement* meEeNoZsRecVsSimE_;

  MonitorElement* meFullRoRu_;
  MonitorElement* meZs1Ru_;
  MonitorElement* meForcedRu_;

  MonitorElement* meLiTtf_;
  MonitorElement* meMiTtf_;
  MonitorElement* meHiTtf_;
  MonitorElement* meForcedTtf_;

  MonitorElement* meTpMap_;

  MonitorElement* meFullRoCnt_;
  MonitorElement* meEbFullRoCnt_;
  MonitorElement* meEeFullRoCnt_;

  MonitorElement* meEbLiZsFir_;
  MonitorElement* meEbHiZsFir_;
  MonitorElement* meEbIncompleteRUZsFir_;

  MonitorElement* meEeLiZsFir_;
  MonitorElement* meEeHiZsFir_;
  MonitorElement* meSRFlagsFromData_;
  MonitorElement* meSRFlagsComputed_;
  MonitorElement* meSRFlagsConsistency_;

  MonitorElement* meIncompleteFRO_;
  MonitorElement* meDroppedFRO_;
  MonitorElement* meCompleteZS_;

  MonitorElement* meIncompleteFROMap_;
  MonitorElement* meDroppedFROMap_;
  MonitorElement* meCompleteZSMap_;

  MonitorElement* meIncompleteFRORateMap_;
  MonitorElement* meDroppedFRORateMap_;
  MonitorElement* meCompleteZSRateMap_;

  MonitorElement* meIncompleteFROCnt_;
  MonitorElement* meDroppedFROCnt_;
  MonitorElement* meCompleteZSCnt_;
  MonitorElement* meEbZsErrCnt_;
  MonitorElement* meEeZsErrCnt_;
  MonitorElement* meZsErrCnt_;
  MonitorElement* meEbZsErrType1Cnt_;
  MonitorElement* meEeZsErrType1Cnt_;
  MonitorElement* meZsErrType1Cnt_;
  //@}

  //@{
  /**Event payload that do not depend on the
   * number of crystals passing the SR
   */
  MonitorElement* meEbFixedPayload_;
  MonitorElement* meEeFixedPayload_;
  MonitorElement* meFixedPayload_;
  //@}

  /** Estimate of L1A rate
   */
  MonitorElement* meL1aRate_;

  ///Counter of FRO-flagged RU dropped from data
  int nDroppedFRO_;

  ///Counter of FRO-flagged RU only partial data
  int nIncompleteFRO_;

  ///Counter of ZS-flagged RU fully read out
  int nCompleteZS_;

  ///Counter of EB FRO-flagged RUs
  int nEbFROCnt_;

  ///Counter of EE FRO-flagged RUs
  int nEeFROCnt_;

  ///Counter of EB ZS errors (LI channel below ZS threshold)
  int nEbZsErrors_;

  ///Counter of EE ZS errors (LI channel below ZS threshold)
  int nEeZsErrors_;

  ///Counter of EB ZS errors of type 1: LI channel below ZS threshold and
  ///in a RU which was fully readout
  int nEbZsErrorsType1_;

  ///Counter of EE ZS errors of tyoe 1: LI channel below ZS threshold and
  ///in a RU which was fully readout
  int nEeZsErrorsType1_;

  /** ECAL trigger tower mapping
   */
  const EcalTrigTowerConstituentsMap* triggerTowerMap_;

  /** Ecal electronics/geometrical mapping.
   */
  const EcalElectronicsMapping* elecMap_;

  /** Local reconstruction switch: true to reconstruct locally the amplitude
   * insted of using the Rec Hits.
   */
  bool localReco_;

  /** Weights for amplitude local reconstruction
   */
  std::vector<double> weights_;

  /** Weights to be used for the ZS FIR filter
   */
  std::vector<int> firWeights_;

  /** ZS threshold in 1/4th ADC count for EB
   */
  int ebZsThr_;

  /** ZS threshold in 1/4th ADC count for EE
   */
  int eeZsThr_;

  /** Switch for uncompressing TP value
   */
  bool tpInGeV_;

  /** Time position of the first sample to use in zero suppession FIR
   * filter. Numbering starts at 0.
   */
  int firstFIRSample_;

  /** Switch to fill histograms with event rate instead of event count.
   * Applies only to some histograms.
   */
  bool useEventRate_;

  /** List of TCC masks for validation
   * If tccMasks[iTcc-1] is false then TCC is considered to have been
   * out of the run and related validations are skipped.
   */
  std::vector<bool> logErrForDccs_;

  /** ECAL barrel read channel count
   */
  int nEb_;

  /** ECAL endcap read channel count
   */
  int nEe_;

  /** ECAL endcap low interest read channel count
   */
  int nEeLI_;

  /** ECAL endcap high interest read channel count
   */
  int nEeHI_;

  /** ECAL barrel low interest read channel count
   */
  int nEbLI_;

  /** ECAL barrel high interest read channel count
   */
  int nEbHI_;

  /** read-out ECAL channel count for each DCC:
   */
  int nPerDcc_[nDccs_];

  /** read-out ECAL Low interest channel count for each DCC:
   */
  int nLiPerDcc_[nDccs_];

  /** read-out ECAL Hiugh interest channel count for each DCC:
   */
  int nHiPerDcc_[nDccs_];

  /** Count for each DCC of RUs with at leat one channel read out:
   */
  int nRuPerDcc_[nDccs_];

  /** Count for each DCC of LI RUs with at leat one channel read out:
   */
  int nLiRuPerDcc_[nDccs_];

  /** Count for each DCC of HI RUs with at leat one channel read out:
   */
  int nHiRuPerDcc_[nDccs_];

  //@{
  /** For book keeping of RU actually read out (not fully zero suppressed)
   */
  bool ebRuActive_[nEbEta / ebTtEdge][nEbPhi / ebTtEdge];
  bool eeRuActive_[nEndcaps][nEeX / scEdge][nEeY / scEdge];
  //@}

  bool isRuComplete_[nDccs_][nDccChs_];

  /** Number of crystal read for each DCC channel (aka readout unit).
   */
  int nPerRu_[nDccs_][nDccChs_];

  /** Event sequence number
   */
  int ievt_;

  /** Trigger tower Et computed as sum the crystal Et. Indices
   * stands for the eta and phi TT index starting from 0 at eta minimum and
   * at phi=0+ in std CMS coordinate system.
   */
  double ttEtSums[nTtEta][nTtPhi];

  /** Energy deposited in ECAL barrel crystals. Eta index starts from 0 at
   * eta minimum and phi index starts at phi=0+ in CMS std coordinate system.
   */
  energiesEb_t ebEnergies[nEbEta][nEbPhi];

  /** Energy deposited in ECAL endcap crystals. Endcap index is 0 for EE- and
   * 1 for EE+. X and Y index starts at x and y minimum in std CMS coordinate
   * system.
   */
  energiesEe_t eeEnergies[nEndcaps][nEeX][nEeY];

  /** Permits to skip inner SC
   */
  bool SkipInnerSC_;

  /** List of enabled histograms. Special name "all" is used to indicate
   * all available histograms.
   */
  std::set<std::string> histList_;

  /** When true, every histogram is enabled.
   */
  bool allHists_;

  /** Histogram directory PATH in DQM or within the output ROOT file
   */
  std::string histDir_;

  /** List of available histograms. Filled by the booking methods.
   * key: name, value: title.
   */
  std::map<std::string, std::string> availableHistList_;

  /** Indicates if EE sim hits are available
   */
  bool withEeSimHit_;

  /** Indicates if EB sim hits are available
   */
  bool withEbSimHit_;

  /** Register a histogram in the available histogram list and check if
   * the histogram is enabled. Called by the histogram booking methods.
   * @return true if the histogram is enable, false otherwise
   */
  bool registerHist(const std::string& name, const std::string& title);

  /** Prints the list of available histograms
   * (registered by the registerHist method), including disabled one.
   */
  void printAvailableHists();

  /** Configure DCC ZS FIR weights. Heuristic is used to determine
   * if input weights are normalized weights or integer weights in
   * the hardware representation.
   * @param weightsForZsFIR weights from configuration file
   */
  void configFirWeights(const std::vector<double>& weightsForZsFIR);

  /** Switch to log in an ascii file inconsistencies found
   * between SRFs read from data and SRFs obtained by rerunning
   * SRP algorithm on TTFs.
   */
  bool logSrpAlgoErrors_;

  /** Switch to log SR decision that fails to be applied on data:
   * inconstitencies between SRF and number of read out crystals.
   */
  bool logSrApplicationErrors_;

  /** Compares two SR flag collection, flags read from data and computed flags.
   * Descripencies are recorded in relevant histogram and log file.
   * @tparam T collection type. Must be either an EESrFlagCollection or an
   * EBSrFlagCollection.
   * @param event event currently analyzed. Used in logs.
   * @param srfFromData SR flag collection read from data
   * @param compareSrf SR flag collection computed from TTF by SRP emulation
   */
  template <class T>
  void compareSrfColl(const edm::Event& event, T& srfFromData, T& computedSrf);

  /** Checks application of SR decision by the DCC.
   * @param event event currently analyzed.
   * @param srfs Selective readou flags
   */
  template <class T>
  void checkSrApplication(const edm::Event& event, T& srfs);

  /** Functions to compute x and y coordinates of RU maps
   * grouping endcap and barrel.
   */
  ///@{
  int ruGraphX(const EcalScDetId& id) const { return id.ix() + (id.zside() > 0 ? 20 : -40); }

  int ruGraphY(const EcalScDetId& id) const { return id.iy(); }

  int ruGraphX(const EcalTrigTowerDetId& id) const { return id.ieta(); }

  int ruGraphY(const EcalTrigTowerDetId& id) const { return id.iphi(); }

  int xtalGraphX(const EEDetId& id) const { return id.ix() + (id.zside() > 0 ? 100 : -200); }

  int xtalGraphY(const EEDetId& id) const { return id.iy(); }

  int xtalGraphX(const EBDetId& id) const { return id.ieta(); }

  int xtalGraphY(const EBDetId& id) const { return id.iphi(); }

  ///@}

  //@{
  /** Retrieves the ID of the DCC reading a readout unit
   * @param detId detid of the readout unit
   */
  int dccId(const EcalScDetId& detId) const;
  int dccId(const EcalTrigTowerDetId& detId) const;
  //@}

  /** Look in events whose DCC has SR flags and
   * enable error logging for them. To be called with
   * the processed first event. List of monitored DCCs
   * is reported in the log file.
   */
  void selectFedsForLog();

  /** Retrieves number of crystal channel read out by a DCC channel
   * @param iDcc DCC ID starting from 1
   * @param iDccCh DCC channel starting from 1
   * @return crystal count
   */
  int getCrystalCount(int iDcc, int iDccCh);
};

#endif  //EcalSelectiveReadoutValidation_H not defined
