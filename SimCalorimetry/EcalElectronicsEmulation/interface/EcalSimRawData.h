/*  
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "SimCalorimetry/EcalSelectiveReadoutAlgos/src/EcalSelectiveReadout.h"
#include <string>
#include <fstream>

/** The EcalSimRawData CMSSW module produces raw data from digis. The raw data
 * are written into files which can be loaded into the TCC DCC and SRP boards
 * in order to emulate the front-end. Only barrel is fully supported.
 * The produced files for TCC assumes a special LUT in the TCC forged for
 * FE emulation mode.
 * <P>Module Parameters:
 * <UL><LI>string digiProducer: digi label</LI>
 *    <LI>string EBDigiCollection: EB crystal digi product instance name</LI>
 *    <LI>string EEDigiCollection: EE crystal digi product instance name</LI>
 *    <LI>string srProducer: sr flags digi label"</LI>
 *    <LI>string EBSrFlagCollection: EB SR flags product instance name</LI>
 *    <LI>string EESrFlagCollection: EE SR flags product instance name</LI>
 *    <LI>string trigPrimProducer: trigger primitive digi label"</LI>
 *    <LI>string trigPrimDigiCollection: trigger primitive digi product instance name</LI>
 *    <LI>string tcpPrimDigiCollection: TCP FENIX output trigger primitive digi product instance name</LI>
 *    <LI>string writeMode: output format. "write", "littleEndian", "bigEndian"
 *    <LI>untracked bool tpVerbose: make verbose the trigger primitive processing</LI>
 *    <LI>untracked bool xtalVerbose: make verbose the crystal digi processing</LI>
 *    <LI>untracked int32 dccNum: Id of the dcc raw data must be produced for. -1 means every DCC</LI>
 *    <LI>untracked int32 tccNum: Id of the tcc raw data must be produced for. -1 means every TCC</LI>
 *    <LI>untracked bool tcc2dccData: switch for TCC->DCC data stream production</LI>
 *    <LI>untracked bool srp2dccData: switch for SRP->DCC data stream production</LI>
 *    <LI>untracked int32 tccInDefaultVal: default TriggerPrimitive values if the trigger tower is abscent</LI>
 *    <LI>untracked string outputBaseName: basename for output files</LI>
 *    </UL>
 */
class EcalSimRawData: public edm::one::EDAnalyzer<> {
  public:
  /** Constructor
   * @param pset CMSSW configuration
   */
  explicit EcalSimRawData(const edm::ParameterSet& pset);

  /** Destructor
   */
  ~EcalSimRawData() override{};

  /** Main method. Called back for each event. This method produced the
   * raw data and write them to disk.
   */
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:

  int iEvent;

  /** Number of crystals in ECAL barrel along eta
   */
  static const int nEbEta = 170;

  /** Number of crystals in ECAL barrel along phi
   */
  static const int nEbPhi = 360;

  /** X-edge of endcap (x,y)- crystal grid
   */
  static const int nEeX = 100;

  /** Y-edge of endcap (x,y)- crystal grid
   */
  static const int nEeY = 100;

  /** Number of endcaps
   */
  static const int nEndcaps = 2;

  /** Supercrystal edge in number of crystals
   */
  static const int scEdge = 5;

  /** Maximum number of supercrystal along x axis
   */
  static const int nScX = 20;

  /** Maximum number of supercrystal along y axis
   */
  static const int nScY = 20;

  /* Edge size of a barrel TT in num. of crystals
  */
  static const int ttEdge = 5;

  /** Number of TTs along SM phi
   */
  static const int nTtSmPhi = 4;

  /** Number of TTs along SM eta
   */
  static const int nTtSmEta = 17;

  /** Number of TTs along Ecal Phi
   */
  static const int nTtPhi = nEbPhi/ttEdge;//72

  /** Number of TTs along Ecal barrel eta
   */
  static const int nEbTtEta = nEbEta/ttEdge;//34

  /** Number of TTs along eta for one endcap.
   */
  static const int nEeTtEta = 11;

  /** Number of TTs along ECAL eta
   */
  static const int nTtEta = nEbTtEta+2*nEeTtEta;//56

  /** Number of barrel DCCs along Phi
   */
  static const int nDccInPhi = 18;

  /** Number of DCCs for a single endcap
   */
  static const int nDccEndcap = 9;

  /** number of TTs along phi of a TCC sector
   */
  static const int ebTccPhiEdge = 20;

  /** Number of Barrel TTCs along phi
   */
  static const int nTccInPhi = 18;

  /** Number of TCCs for a single endcap
   */
  static const int nTccEndcap = 36;

  /** Number of barrel crystals along phi covered by a DCC
   */
  static const int ebDccPhiEdge = 20;

  /** Number of trigger towers alng phi covered by a DCC
   */ 
  static const int nTtPhisPerEbDcc = 4;

  /** Number of trigger towers alng phi covered by a TCC
   */ 
  static const int nTtPhisPerEbTcc = 4;

  /** Number of barrel trigger tower types (in term of VFE card orientation)
   */
  static const int nTtTypes = 2;

  /** Map of trigger tower types (in term of VFE card orientation).
   * ttType[iTtEta0]: trigger tower type of TTs with eta 'c-arrary' index
   * iTtEta0
   */
  static const int ttType[nEbTtEta];

  /** Maps (strip_id, channel_id) to phi index within a TT.
   * stripCh2Phi[iTtType][strip_id-1][ch_id-1] will be the phi index of the
   * TT of type iTtType with strip ID 'strip_id' and channel VFE id 'ch_id'.
   * The phi runs from 0 to 4 and increases with the phi of std CMS
   * coordinates.
   */
  static const int stripCh2Phi[nTtTypes][ttEdge][ttEdge];

  /** Maps strip_id to eta index within a TT.
   * strip2Eta[iTtType][strip_id-1] will be the eta index of channels
   * with strip_id 'strip_id'. The eta index runs from 0 to 4 and increasing
   * with the eta of the std CMS coordinates.
   */
  static const int strip2Eta[nTtTypes][ttEdge];

  /** Output format mode
   * <UL><I>littleEndian: little endian binary</I>
   *     <I>bigEndian: big endian binary</I>
   *     <I>ascii: ascii mode. The one accepted by the TCC, DCC and SRP board
   * control software.</I>
   * </UL>
   */
  enum writeMode_t {littleEndian, bigEndian, ascii};
  
private:
  /*
  const EBDigiCollection*
  getEBDigis(const edm::Event& event) const;
  
  const EEDigiCollection*
  getEEDigis(const edm::Event& event) const;

  const EcalTrigPrimDigiCollection*
  getTrigPrims(const edm::Event& event) const;
  */
  
  /// call these once an event, to make sure everything
  /// is up-to-date
  void
  checkGeometry(const edm::EventSetup& eventSetup);
  void
  checkTriggerMap(const edm::EventSetup& eventSetup);

  /** Converts std CMSSW crystal eta index into a c-index (contiguous integer
   * starting from 0 and increasing with pseudo-rapidity).
   * @param iEta std CMSSW crystal eta index
   * @return the c-array index
   */
  int iEta2cIndex(int iEta) const{
    return (iEta<0)?iEta+85:iEta+84;
  }

  /** Converts std CMSSW crystal phi index into a c-index (contiguous integer
   * starting from 0 at phi=0deg and increasing with phi).
   * @param iPhi std CMSSW crystal phi index
   * @return the c-array index
   */
  int iPhi2cIndex(int iPhi) const{
    int iPhi0 = iPhi -11;
    if(iPhi0<0) iPhi0+=nEbPhi;
    return iPhi0;
  }

  /** Converts std CMSSW ECAL trigger tower eta index into
   * a c-index (contiguous integer starting from 0 and increasing with
   * pseudo-rapidity).
   * @param iEta std CMSSW trigger tower eta index
   * @return the c-array index
   */
  int iTtEta2cIndex(int iTtEta) const{
    return (iTtEta<0)?(iTtEta+28):(iTtEta+27);
  }

  /** Converse of iTtEta2cIndex
   * @param iTtEta0 c eta index of TT
   * @param std CMSSW TT eta index
   */
  int cIndex2iTtEta(int iTtEta0) const{
    return (iTtEta0<28)?(iTtEta0-28):(iTtEta0-27);
  }

  /** Converse of iTtPhi2cIndex
   * @param iTtPhi0 phi index of TT
   * @return std CMSS TT index
   */
  int cIndex2TtPhi(int iTtPhi0) const{
    return iTtPhi0+1;
  }
  
  /** Converts std CMSSW ECAL trigger tower phi index into a c-index
   * (contiguous integer starting from 0 at phi=0deg and increasing with phi).
   * @param iPhi std CMSSW ECAL trigger tower phi index
   * @return the c-array index
   */
  int iTtPhi2cIndex(int iTtPhi) const{
    return iTtPhi-1;
  }

  /*
    int iXY2cIndex(int iX) const{
    return iX-1;
  }
  */

  /** Converts electronic number of an ECAL barrel channel to geometrical
   * indices
   * @param ittEta0 trigger tower c index
   * @param ittPhi0 trigger tower c index
   * @param strip1 strip index within the TT. Runs from 1 to 5.
   * @param ch1 channel electronics number within the VFE. Runs from 1 to 5.
   * @param [out] iEta0 eta c index of the channel
   * @param [out] iPhi0 eta c index of the channel
   */
  void elec2GeomNum(int ittEta0, int ittPhi0, int strip1,
		    int ch1, int& iEta0, int& iPhi0) const;

  /* Set horizontal parity of a 16-bit word of FE data
   * @param a the word whose parity must be set.
   */
  void setHParity(uint16_t& a) const;

  /** Generates FE crystal data
   * @param basename base for the output file name. DCC number is appended to
   * the name
   * @param iEvent event index
   * @param adcCount the payload, the ADC count of the channels.
   */
  void genFeData(std::string basename, int iEvent,
		 const std::vector<uint16_t> adcCount[nEbEta][nEbPhi]) const;


  /** Generates FE trigger primitives data
   * @param basename base for the output file name. DCC number is appended to
   * the name
   * @param iEvent event index
   * @param tps the payload, the trigger primitives
   */
  void genTccIn(std::string basename, int iEvent,
		const int tps[nTtEta][nTtPhi]) const;

  /** Generates TCC->DCC data
   * @param basename base for the output file name. DCC number is appended to
   * the name
   * @param iEvent event index
   * @param tps the payload, the trigger primitives
   */
  void genTccOut(std::string basename, int iEvent,
		 const int tps[nTtEta][nTtPhi]) const;


  
  /** Retrieves barrel digis (APD ADC count).
   * @param event CMS event
   * @param adc [out] the adc counts: adc[iEta0][iPhi0][iTimeSample0] 
   */
  void getEbDigi(const edm::Event& event,
		 std::vector<uint16_t> adc[nEbEta][nEbPhi]) const;


  /** Extracts the trigger primitive (TP). The collection name
   * parameter permits to select either the TCP Fenix output
   * or the TCC output.
   * Note: TCP output is only valid for the barrel.
   * @param event CMS event
   * @param collName label of the EDM collection containing the TP.
   * @param tp [out] the trigger primitives
   */
  void getTp(const edm::Event& event, const std::string& collName,
	     int tp[nTtEta][nTtPhi]) const;

  
  /** Help function to get the file extension which depends on the output
   * formats.
   */
  std::string getExt() const;

  /** Write a data 16-bit word into file according to selected format.
   * @param f the file stream to write to
   * @param data the peace of data to write
   * @param [in,out] iword pass zero when writing for the first time in a file,
   * then the value returned by the previous call. Counts the number of words
   * written into the file.
   * @param hpar if true the horizontal odd word parity is set before writing
   * the word into the file.
   */
  void fwrite(std::ofstream& f, uint16_t data, int& iword,
	      bool hpar = true) const;


  /** Computes the selective readout flags.
   * @param [in] event CMS event
   * @param [out] ebSrf the computed SR flags for barrel
   * @param [out] eeSrf the computed SR flags for endcaps
   */
  void getSrfs(const edm::Event& event, int ebSrf[nTtEta][nTtPhi],
	       int eeSrf[nEndcaps][nScX][nScY]) const;

  /** Generates SR flags
   * @param basename base for the output file name. DCC number is appended to
   * the name
   * @param iEvent event index
   * @param the trigger tower flags
   */
  void genSrData(std::string basename, int iEvent,
		 int ttf[nEbTtEta][nTtPhi]) const;
  
private:
  /** Name of module/plugin/producer making digis
   */
  std::string digiProducer_;

  /** EB digi product instance name
   */
  std::string ebDigiCollection_;

  /** EE digi product instance name
   */
  std::string eeDigiCollection_;


  /** Label of SR flags and suppressed digis
   */
  std::string srDigiProducer_;
  
   /** EB SRP flag digi product instance name
   */
  std::string ebSrFlagCollection_;

  /** EE SRP flag digi product instancename
   */
  std::string eeSrFlagCollection_;

  /** Trigger primitive digi product instance name
   */
  std::string tpDigiCollection_; 

  /** TCP Fenix output digi product instance name
   */
  std::string tcpDigiCollection_; 
  
  /** Calorimeter geometry
   */
  const CaloGeometry * theGeometry;

  /** Name of the trigger primitive label
   */
  std::string tpProducer_;

  /** output format
   */
  writeMode_t writeMode_;

  /** Verbosity switch for crystal data
   */
  bool xtalVerbose_;

  /** Verbosity switch for crystal data
   */
  bool tpVerbose_;

  /** Switch for data of SRP->DCC link
   */
  bool srp2dcc_;

  /** Switch for data of TCC->DCC link
   */
  bool tcc2dcc_;

  /** Switch for data of FE->DCC link
   */
  bool fe2dcc_;

  /** Switch for data of FE->TCC link
   */
  bool fe2tcc_;

  /** ECAL endcap trigger tower map
   */
  const EcalTrigTowerConstituentsMap * theTriggerTowerMap;

  /** Selective readout simulator
   */
  std::unique_ptr<EcalSelectiveReadout> esr_;

  /** Output file for trigger tower flags
   */
  std::ofstream ttfFile;

  /** Output file for selective readout flags
   */
  std::ofstream srfFile;

  /** Index of the TCC, FE data must be produced for. -1 for all TTCs
   */
  int tccNum_;
  
  /** Index of the DCC, FE data must be produced for. -1 for all TTCs
   */
  int dccNum_;
  
  /** default TriggerPrimitive values if the trigger tower is abscent
   */
  int tccInDefaultVal_;
  
  /** basename for output files
   */
  std::string basename_;
};

