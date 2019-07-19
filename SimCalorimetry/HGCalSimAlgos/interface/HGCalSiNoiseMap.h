#ifndef simcalorimetry_hgcalsimalgos_hgcalsinoisemap
#define simcalorimetry_hgcalsimalgos_hgcalsinoisemap

#include "SimCalorimetry/HGCalSimAlgos/interface/HGCalRadiationMap.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include <string>

/**
   @class HGCalSiNoiseMap
   @short derives from HGCalRadiation map to parse fluence parameters, provides Si-specific functions
*/
class HGCalSiNoiseMap : public HGCalRadiationMap {
public:
  enum GainRange_t { q80fC, q160fC, q320fC, AUTO };

  struct SiCellOpCharacteristics {
    SiCellOpCharacteristics()
        : lnfluence(0.), fluence(0.), ileak(0.), cce(1.), noise(0.), mipfC(0), gain(0), mipADC(0), thrADC(0) {}
    double lnfluence, fluence, ileak, cce, noise, mipfC;
    unsigned int gain, mipADC, thrADC, maxADC;
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
    cceParam_[HGCSiliconDetId::waferType::HGCalFine] = parsFine;          //120
    cceParam_[HGCSiliconDetId::waferType::HGCalCoarseThin] = parsThin;    //200
    cceParam_[HGCSiliconDetId::waferType::HGCalCoarseThick] = parsThick;  //300
  }

  /**
     @short returns the charge collection efficiency and noise
     if gain range is set to auto, it will find the most appropriate gain to put the mip peak close to 10 ADC counts
  */
  SiCellOpCharacteristics getSiCellOpCharacteristics(const HGCSiliconDetId &did,
                                                     GainRange_t gain = GainRange_t::AUTO,
                                                     bool ignoreFluence = false,
                                                     int aimMIPtoADC = 10);

  std::map<HGCSiliconDetId::waferType, double> &getMipEqfC() { return mipEqfC_; }
  std::map<HGCSiliconDetId::waferType, double> &getCellCapacitance() { return cellCapacitance_; }
  std::map<HGCSiliconDetId::waferType, double> &getCellVolume() { return cellVolume_; }
  std::map<HGCSiliconDetId::waferType, std::vector<double> > &getCCEParam() { return cceParam_; }
  std::vector<double> &getIleakParam() { return ileakParam_; }
  std::map<GainRange_t, std::vector<double> > &getENCsParam() { return encsParam_; }
  std::map<GainRange_t, double> &getLSBPerGain() { return lsbPerGain_; }
  std::map<GainRange_t, double> &getMaxADCPerGain() { return maxADCPerGain_; }

private:
  //
  std::map<HGCSiliconDetId::waferType, double> mipEqfC_, cellCapacitance_, cellVolume_;
  std::map<HGCSiliconDetId::waferType, std::vector<double> > cceParam_;

  //leakage current/volume vs fluence
  std::vector<double> ileakParam_;

  //shaper noise param
  const double encpScale_;

  //common noise subtraction noise (final scaling value)
  const double encCommonNoiseSub_;

  //electron charge in fC
  const double qe2fc_;

  //electronics noise (series+parallel) polynomial coeffs;
  std::map<GainRange_t, std::vector<double> > encsParam_;

  //lsb
  std::map<GainRange_t, double> lsbPerGain_, maxADCPerGain_;
};

#endif
