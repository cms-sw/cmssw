#ifndef simcalorimetry_hgcalsimalgos_hgcalsinoisemap
#define simcalorimetry_hgcalsimalgos_hgcalsinoisemap

#include "SimCalorimetry/HGCalSimAlgos/interface/HGCalRadiationMap.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include <string>
#include <array>

/**
   @class HGCalSiNoiseMap
   @short derives from HGCalRadiation map to parse fluence parameters, provides Si-specific functions; see DN-19-045
*/
class HGCalSiNoiseMap : public HGCalRadiationMap {
public:
  enum GainRange_t { q80fC, q160fC, q320fC, AUTO };
  enum NoiseMapAlgoBits_t { FLUENCE, CCE, NOISE };

  struct SiCellOpCharacteristics {
    SiCellOpCharacteristics()
        : lnfluence(0.), fluence(0.), ileak(0.), cce(1.), noise(0.), mipfC(0), gain(0), mipADC(0), thrADC(0) {}
    double lnfluence, fluence, ileak, cce, noise, mipfC;
    unsigned int gain, mipADC, thrADC;
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
     @short returns the charge collection efficiency and noise
     if gain range is set to auto, it will find the most appropriate gain to put the mip peak close to 10 ADC counts
  */
  SiCellOpCharacteristics getSiCellOpCharacteristics(const HGCSiliconDetId &did,
                                                     GainRange_t gain = GainRange_t::AUTO,
                                                     int aimMIPtoADC = 10);

  std::array<double, 3> &getMipEqfC() { return mipEqfC_; }
  std::array<double, 3> &getCellCapacitance() { return cellCapacitance_; }
  std::array<double, 3> &getCellVolume() { return cellVolume_; }
  std::vector<std::vector<double> > &getCCEParam() { return cceParam_; }
  std::vector<double> &getIleakParam() { return ileakParam_; }
  std::vector<std::vector<double> > &getENCsParam() { return encsParam_; }
  std::vector<double> &getLSBPerGain() { return lsbPerGain_; }
  std::vector<double> &getMaxADCPerGain() { return chargeAtFullScaleADCPerGain_; }

private:
  //vector of three params, per sensor type: 0:120 [mum], 1:200, 2:300
  std::array<double, 3> mipEqfC_, cellCapacitance_, cellVolume_;
  std::vector<std::vector<double> > cceParam_;

  //leakage current/volume vs fluence
  std::vector<double> ileakParam_;

  //shaper noise param
  const double encpScale_;

  //common noise subtraction noise (final scaling value)
  const double encCommonNoiseSub_;

  //electron charge in fC
  const double qe2fc_;

  //electronics noise (series+parallel) polynomial coeffs;
  std::vector<std::vector<double> > encsParam_;

  //lsb
  std::vector<double> lsbPerGain_, chargeAtFullScaleADCPerGain_;

  //conversions
  const double unitToMicro_ = 1.e6;

  //flags used to disable specific components of the Si operation parameters
  bool ignoreFluence_, ignoreCCE_, ignoreNoise_;
};

#endif
