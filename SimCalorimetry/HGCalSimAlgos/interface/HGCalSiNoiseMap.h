#ifndef simcalorimetry_hgcalsimalgos_hgcalsinoisemap
#define simcalorimetry_hgcalsimalgos_hgcalsinoisemap

#include "SimCalorimetry/HGCalSimAlgos/interface/HGCalRadiationMap.h"
#include "SimCalorimetry/HGCalSimProducers/interface/HGCFEElectronics.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/ForwardDetId/interface/HFNoseDetId.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include <string>
#include <array>
#include <unordered_map>

/**
   @class HGCalSiNoiseMap
   @short derives from HGCalRadiation map to parse fluence parameters, provides Si-specific functions; see DN-19-045
*/
template <typename T>
class HGCalSiNoiseMap : public HGCalRadiationMap {
public:
  enum GainRange_t { q80fC, q160fC, q320fC, AUTO };
  enum NoiseMapAlgoBits_t { FLUENCE, CCE, NOISE, PULSEPERGAIN, CACHEDOP };

  struct SiCellOpCharacteristicsCore {
    SiCellOpCharacteristicsCore() : cce(0.), noise(0.), gain(0), thrADC(0) {}
    float cce, noise;
    unsigned short gain, thrADC;
  };

  struct SiCellOpCharacteristics {
    SiCellOpCharacteristics() : lnfluence(0.), fluence(0.), ileak(0.), enc_s(0.), enc_p(0.), mipfC(0), mipADC(0) {}
    SiCellOpCharacteristicsCore core;
    double lnfluence, fluence, ileak, enc_s, enc_p, mipfC;
    unsigned int mipADC;
  };

  HGCalSiNoiseMap();
  ~HGCalSiNoiseMap(){};

  /**
     @short set the ileak parameters to use
  */
  void setIleakParam(const std::vector<double> &pars) { ileakParam_ = pars; }

  /**
     @short set the cce parameters to use
  */
  void setCceParam(const std::vector<double> &parsFine,
                   const std::vector<double> &parsThin,
                   const std::vector<double> &parsThick) {
    cceParam_.push_back(parsFine);   //120
    cceParam_.push_back(parsThin);   //200
    cceParam_.push_back(parsThick);  //300
  }

  /**
     @short overrides base class method with specifics for the configuration of the algo
  */
  void setDoseMap(const std::string &, const unsigned int &);

  /**
     @short specialization of the base class method which sets the geometry so that it can instantiate an operation
     cache the first time it is called - intrinsically related to the valid detIds in the geometry
     the filling of the cache is ignored by configuration or if it has already been filled
   */
  void setGeometry(const CaloSubdetectorGeometry *, GainRange_t gain = GainRange_t::AUTO, int aimMIPtoADC = 10);

  /**
     @short returns the charge collection efficiency and noise
     if gain range is set to auto, it will find the most appropriate gain to put the mip peak close to 10 ADC counts
  */
  const SiCellOpCharacteristicsCore getSiCellOpCharacteristicsCore(const T &did, GainRange_t gain, int aimMIPtoADC);
  const SiCellOpCharacteristicsCore getSiCellOpCharacteristicsCore(const T &did) {
    return getSiCellOpCharacteristicsCore(did, defaultGain_, defaultAimMIPtoADC_);
  }

  SiCellOpCharacteristics getSiCellOpCharacteristics(const T &did,
                                                     GainRange_t gain = GainRange_t::AUTO,
                                                     int aimMIPtoADC = 10);
  SiCellOpCharacteristics getSiCellOpCharacteristics(double &cellCap,
                                                     double &cellVol,
                                                     double &mipEqfC,
                                                     std::vector<double> &cceParam,
                                                     int &subdet,
                                                     int &layer,
                                                     double &radius,
                                                     GainRange_t &gain,
                                                     int &aimMIPtoADC);

  std::array<double, 3> &getMipEqfC() { return mipEqfC_; }
  std::array<double, 3> &getCellCapacitance() { return cellCapacitance_; }
  std::array<double, 3> &getCellVolume() { return cellVolume_; }
  std::vector<std::vector<double> > &getCCEParam() { return cceParam_; }
  std::vector<double> &getIleakParam() { return ileakParam_; }
  std::vector<std::vector<double> > &getENCsParam() { return encsParam_; }
  std::vector<double> &getLSBPerGain() { return lsbPerGain_; }
  void setDefaultADCPulseShape(const hgc_digi::FEADCPulseShape &adcPulse) { defaultADCPulse_ = adcPulse; };
  const hgc_digi::FEADCPulseShape &adcPulseForGain(GainRange_t gain) {
    if (ignoreGainDependentPulse_)
      return defaultADCPulse_;
    return adcPulses_[gain];
  };
  std::vector<double> &getMaxADCPerGain() { return chargeAtFullScaleADCPerGain_; }
  double getENCpad(double ileak);
  void setCachedOp(bool flag) { activateCachedOp_ = flag; }

  inline void setENCCommonNoiseSubScale(double val) { encCommonNoiseSub_ = val; }

private:
  GainRange_t defaultGain_;
  int defaultAimMIPtoADC_;

  //cache of SiCellOpCharacteristics
  std::map<uint32_t, SiCellOpCharacteristicsCore> siopCache_;

  //vector of three params, per sensor type: 0:120 [mum], 1:200, 2:300
  std::array<double, 3> mipEqfC_, cellCapacitance_, cellVolume_;
  std::vector<std::vector<double> > cceParam_;

  //leakage current/volume vs fluence
  std::vector<double> ileakParam_;

  //common noise subtraction noise (final scaling value)
  double encCommonNoiseSub_;

  //electron charge in fC
  const double qe2fc_;

  //electronics noise (series+parallel) polynomial coeffs and ADC pulses;
  std::vector<std::vector<double> > encsParam_;
  hgc_digi::FEADCPulseShape defaultADCPulse_;
  std::vector<hgc_digi::FEADCPulseShape> adcPulses_;

  //lsb
  std::vector<double> lsbPerGain_, chargeAtFullScaleADCPerGain_;

  //conversions
  const double unitToMicro_ = 1.e6;
  const double unitToMicroLog_ = log(unitToMicro_);

  //flags used to disable specific components of the Si operation parameters or usage of operation cache
  bool ignoreFluence_, ignoreCCE_, ignoreNoise_, ignoreGainDependentPulse_, activateCachedOp_;
};

template class HGCalSiNoiseMap<HGCSiliconDetId>;
template class HGCalSiNoiseMap<HFNoseDetId>;

#endif
