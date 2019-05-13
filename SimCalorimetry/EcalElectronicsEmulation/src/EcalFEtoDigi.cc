#include "CondFormats/DataRecord/interface/EcalTPGLutGroupRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGLutIdMapRcd.h"
#include "CondFormats/EcalObjects/interface/EcalTPGLutGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGLutIdMap.h"
#include "SimCalorimetry/EcalElectronicsEmulation/interface/EcalFEtoDigi.h"

EcalFEtoDigi::EcalFEtoDigi(const edm::ParameterSet &iConfig) {
  basename_ = iConfig.getUntrackedParameter<std::string>("FlatBaseName", "ecal_tcc_");
  sm_ = iConfig.getUntrackedParameter<int>("SuperModuleId", -1);
  fileEventOffset_ = iConfig.getUntrackedParameter<int>("FileEventOffset", 0);
  useIdentityLUT_ = iConfig.getUntrackedParameter<bool>("UseIdentityLUT", false);
  debug_ = iConfig.getUntrackedParameter<bool>("debugPrintFlag", false);

  singlefile = (sm_ == -1) ? false : true;

  produces<EcalTrigPrimDigiCollection>();
  produces<EcalTrigPrimDigiCollection>("formatTCP");
}

/// method called to produce the data
void EcalFEtoDigi::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  /// event counter
  static int current_bx = -1;
  current_bx++;

  /// re-read input (needed in case of event-by-event input production)
  // readInput();

  if (debug_)
    std::cout << "[EcalFEtoDigi::produce] producing event " << current_bx << std::endl;

  std::unique_ptr<EcalTrigPrimDigiCollection> e_tpdigis(new EcalTrigPrimDigiCollection);
  std::unique_ptr<EcalTrigPrimDigiCollection> e_tpdigisTcp(new EcalTrigPrimDigiCollection);

  std::vector<TCCinput>::const_iterator it;

  for (int i = 0; i < N_SM; i++) {
    if (!singlefile)
      sm_ = i + 1;

    for (it = inputdata_[i].begin(); it != inputdata_[i].end(); it++) {
      if (!(*it).is_current(current_bx + fileEventOffset_))
        continue;
      else if (debug_ && (*it).input != 0)
        std::cout << "[EcalFEtoDigi] "
                  << "\tsupermodule:" << sm_ << "\tevent: " << current_bx << "\tbx: " << (*it).bunchCrossing
                  << "\tvalue:0x" << std::setfill('0') << std::setw(4) << std::hex << (*it).input << std::setfill(' ')
                  << std::dec << std::endl;

      /// create EcalTrigTowerDetId
      const EcalTrigTowerDetId e_id = create_TTDetId(*it);

      // EcalElectronicsMapping theMapping;
      // const EcalTrigTowerDetId  e_id
      //= theMapping.getTrigTowerDetId(SMidToTCCid(sm_),(*it).tower);
      // EcalElectronicsMapping::getTrigTowerDetId(int TCCid, int iTT)

      /// create EcalTriggerPrimitiveDigi
      EcalTriggerPrimitiveDigi *e_digi = new EcalTriggerPrimitiveDigi(e_id);
      EcalTriggerPrimitiveDigi *e_digiTcp = new EcalTriggerPrimitiveDigi(e_id);

      /// create EcalTriggerPrimitiveSample
      EcalTriggerPrimitiveSample e_sample = create_TPSample(*it, iSetup);
      EcalTriggerPrimitiveSample e_sampleTcp = create_TPSampleTcp(*it, iSetup);

      /// set sample
      e_digi->setSize(1);  // set sampleOfInterest to 0
      e_digi->setSample(0, e_sample);

      /// add to EcalTrigPrimDigiCollection
      e_tpdigis->push_back(*e_digi);

      /// set sample (uncompressed format)
      e_digiTcp->setSize(1);  // set sampleOfInterest to 0
      e_digiTcp->setSample(0, e_sampleTcp);

      /// add to EcalTrigPrimDigiCollection (uncompressed format)
      e_tpdigisTcp->push_back(*e_digiTcp);

      if (debug_)
        outfile << (*it).tower << '\t' << (*it).bunchCrossing << '\t' << std::setfill('0') << std::hex << "0x"
                << std::setw(4) << (*it).input << '\t' << "0" << std::dec << std::setfill(' ') << std::endl;

      /// print & debug
      if (debug_ && (*it).input != 0)
        std::cout << "[EcalFEtoDigi] debug id: " << e_digi->id() << "\n\t" << std::dec
                  << "\tieta: " << e_digi->id().ieta() << "\tiphi: " << e_digi->id().iphi()
                  << "\tsize: " << e_digi->size() << "\tfg: " << (e_digi->fineGrain() ? 1 : 0) << std::hex << "\tEt: 0x"
                  << e_digi->compressedEt() << " (0x" << (*it).get_energy() << ")"
                  << "\tttflag: 0x" << e_digi->ttFlag() << std::dec << std::endl;

      delete e_digi;
      delete e_digiTcp;
    }

    if (singlefile)
      break;
  }

  /// in case no info was found for the event:need to create something
  if (e_tpdigis->empty()) {
    std::cout << "[EcalFEtoDigi] creating empty collection for the event!\n";
    EcalTriggerPrimitiveDigi *e_digi = new EcalTriggerPrimitiveDigi();
    e_tpdigis->push_back(*e_digi);
  }

  iEvent.put(std::move(e_tpdigis));
  iEvent.put(std::move(e_tpdigisTcp), "formatTCP");
}

/// open and read in input (flat) data file
void EcalFEtoDigi::readInput() {
  if (debug_)
    std::cout << "\n[EcalFEtoDigi::readInput] Reading input data\n";

  if (!singlefile)
    sm_ = -1;
  for (int i = 0; i < N_SM; i++)
    inputdata_[i].clear();

  std::stringstream s;
  int tcc;

  for (int i = 0; i < N_SM; i++) {
    tcc = (sm_ == -1) ? SMidToTCCid(i + 1) : SMidToTCCid(sm_);

    s.str("");
    s << basename_ << tcc << ".txt";

    std::ifstream f(s.str().c_str());

    if (debug_) {
      std::cout << "  opening " << s.str().c_str() << "..." << std::endl;
      if (!f.good())
        std::cout << " skipped!";
      std::cout << std::endl;
    }
    // if (!f.good() || f.eof())
    //  throw cms::Exception("BadInputFile")
    //	<< "EcalFEtoDigi: cannot open file " << s.str().c_str() << std::endl;

    int n_bx = 0;
    int tt;
    int bx;
    unsigned val;
    int dummy;

    while (f.good()) {
      if (f.eof())
        break;
      tt = 0;
      bx = -1;
      val = 0x0;
      dummy = 0;
      f >> tt >> bx >> std::hex >> val >> std::dec >> dummy;
      if (bx == -1 || bx < fileEventOffset_)
        continue;
      if (!n_bx || (bx != (inputdata_[i].back()).bunchCrossing))
        n_bx++;
      TCCinput ttdata(tt, bx, val);
      inputdata_[i].push_back(ttdata);

      if (debug_ && val != 0)
        printf("\treading tower:%d  bx:%d input:0x%x dummy:%2d\n", tt, bx, val, dummy);
    }

    f.close();

    if (sm_ != -1)
      break;
  }

  if (debug_)
    std::cout << "[EcalFEtoDigi::readInput] Done reading." << std::endl;

  return;
}

/// create EcalTrigTowerDetId from input data (line)
EcalTrigTowerDetId EcalFEtoDigi::create_TTDetId(TCCinput data) {
  // (EcalBarrel only)
  static const int kTowersInPhi = 4;

  int iTT = data.tower;
  int zside = (sm_ > 18) ? -1 : +1;
  int SMid = sm_;

  int jtower = iTT - 1;
  int etaTT = jtower / kTowersInPhi + 1;
  int phiTT;
  if (zside < 0)
    phiTT = (SMid - 19) * kTowersInPhi + jtower % kTowersInPhi;
  else
    phiTT = (SMid - 1) * kTowersInPhi + kTowersInPhi - (jtower % kTowersInPhi) - 1;

  phiTT++;
  // needed as phi=0 (iphi=1) is at middle of lower SMs (1 and 19), need shift
  // by 2
  phiTT = phiTT - 2;
  if (phiTT <= 0)
    phiTT = 72 + phiTT;

  /// construct the EcalTrigTowerDetId object
  if (debug_ && data.get_energy() != 0)
    printf(
        "[EcalFEtoDigi] Creating EcalTrigTowerDetId "
        "(SMid,itt)=(%d,%d)->(eta,phi)=(%d,%d) \n",
        SMid,
        iTT,
        etaTT,
        phiTT);

  EcalTrigTowerDetId e_id(zside, EcalBarrel, etaTT, phiTT, 0);

  return e_id;
}

/// create EcalTriggerPrimitiveSample from input data (line)
EcalTriggerPrimitiveSample EcalFEtoDigi::create_TPSample(TCCinput data, const edm::EventSetup &evtSetup) {
  int tower = create_TTDetId(data).rawId();
  int Et = data.get_energy();
  bool tt_fg = data.get_fg();
  // unsigned input = data.input;
  // int  Et    = input & 0x3ff; //get bits 0-9
  // bool tt_fg = input & 0x400; //get bit number 10

  /// setup look up table
  unsigned int lut_[1024];
  if (!useIdentityLUT_)
    getLUT(lut_, tower, evtSetup);
  else
    for (int i = 0; i < 1024; i++)
      lut_[i] = i;  // identity lut!

  /// compress energy 10 -> 8  bit
  int lut_out = lut_[Et];
  int ttFlag = (lut_out & 0x700) >> 8;
  int cEt = (lut_out & 0xff);

  /// crate sample
  if (debug_ && data.get_energy() != 0)
    printf(
        "[EcalFEtoDigi] Creating sample; input:0x%X (Et:0x%x) cEt:0x%x "
        "fg:%d ttflag:0x%x \n",
        data.input,
        Et,
        cEt,
        tt_fg,
        ttFlag);

  EcalTriggerPrimitiveSample e_sample(cEt, tt_fg, ttFlag);

  return e_sample;
}

/// create EcalTriggerPrimitiveSample in tcp format (uncomrpessed energy)
EcalTriggerPrimitiveSample EcalFEtoDigi::create_TPSampleTcp(TCCinput data, const edm::EventSetup &evtSetup) {
  int tower = create_TTDetId(data).rawId();
  int Et = data.get_energy();
  bool tt_fg = data.get_fg();

  /// setup look up table
  unsigned int lut_[1024];
  if (!useIdentityLUT_)
    getLUT(lut_, tower, evtSetup);
  else
    for (int i = 0; i < 1024; i++)
      lut_[i] = i;  // identity lut!

  int lut_out = lut_[Et];
  int ttFlag = (lut_out & 0x700) >> 8;
  int tcpdata = ((ttFlag & 0x7) << 11) | ((tt_fg & 0x1) << 10) | (Et & 0x3ff);

  EcalTriggerPrimitiveSample e_sample(tcpdata);

  return e_sample;
}

/// method called once each job just before starting event loop
void EcalFEtoDigi::beginJob() {
  /// check SM numbering convetion: 1-38
  /// [or -1 flag to indicate all sm's are to be read in]
  if (sm_ != -1 && (sm_ < 1 || sm_ > 36))
    throw cms::Exception("EcalFEtoDigiInvalidDetId") << "EcalFEtoDigi: Adapt SM numbering convention.\n";

  /// debug: open file for recreating input copy
  if (debug_)
    outfile.open("inputcopy.txt");

  readInput();
}

/// method called once each job just after ending the event loop
void EcalFEtoDigi::endJob() {
  if (outfile.is_open())
    outfile.close();
}

/// translate input supermodule id into TCC id (barrel)
int EcalFEtoDigi::SMidToTCCid(const int smid) const { return (smid <= 18) ? smid + 55 - 1 : smid + 37 - 19; }

/// return the LUT from eventSetup
void EcalFEtoDigi::getLUT(unsigned int *lut, const int towerId, const edm::EventSetup &evtSetup) const {
  edm::ESHandle<EcalTPGLutGroup> lutGrpHandle;
  evtSetup.get<EcalTPGLutGroupRcd>().get(lutGrpHandle);
  const EcalTPGGroups::EcalTPGGroupsMap &lutGrpMap = lutGrpHandle.product()->getMap();
  EcalTPGGroups::EcalTPGGroupsMapItr itgrp = lutGrpMap.find(towerId);
  uint32_t lutGrp = 999;
  if (itgrp != lutGrpMap.end())
    lutGrp = itgrp->second;

  edm::ESHandle<EcalTPGLutIdMap> lutMapHandle;
  evtSetup.get<EcalTPGLutIdMapRcd>().get(lutMapHandle);
  const EcalTPGLutIdMap::EcalTPGLutMap &lutMap = lutMapHandle.product()->getMap();
  EcalTPGLutIdMap::EcalTPGLutMapItr itLut = lutMap.find(lutGrp);
  if (itLut != lutMap.end()) {
    const unsigned int *theLut = (itLut->second).getLut();
    for (unsigned int i = 0; i < 1024; i++)
      lut[i] = theLut[i];
  }
}
