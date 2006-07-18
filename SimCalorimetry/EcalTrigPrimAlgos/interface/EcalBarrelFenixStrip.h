#ifndef ECAL_BARREL_FENIXSTRIP_H
#define ECAL_BARREL_FENIXSTRIP_H

#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixChip.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixLinearizer.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixAmplitudeFilter.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixStripFormat.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixPeakFinder.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixEtStrip.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixLinearizer.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixEtStrip.h>


class TTree;
class EBDataFrame;
class EcalTriggerPrimitiveSample;

  /** 
      \class EcalBarrelFenixStrip
      \brief class representing the Fenix chip, format strip, for the barrel
  */

  class EcalBarrelFenixStrip : public EcalFenixChip {

  private:

    EcalFenixLinearizer *linearizer_[nCrystalsPerStrip_];

    EcalFenixAmplitudeFilter *amplitude_filter_; 

    EcalFenixPeakFinder *peak_finder_; 

    EcalFenixStripFormat *fenix_format_;

  public:

    // constructor, destructor
    EcalBarrelFenixStrip(const TTree *tree);
    virtual ~EcalBarrelFenixStrip() ;

    // main methods
    std::vector<int> process(std::vector<const EBDataFrame *> &, int stripnr,int townr);

    // getters for the algorithms
    EcalFenixLinearizer *getLinearizer (int i) const { return linearizer_[i];}
    EcalFenixEtStrip *getAdder() const { return  dynamic_cast<EcalFenixEtStrip *>(adder_);}
    EcalFenixAmplitudeFilter *getFilter() const { return amplitude_filter_;}
    EcalFenixPeakFinder *getPeakFinder() const { return peak_finder_;}
    EcalFenixStripFormat *getFormatter() const { return dynamic_cast<EcalFenixStripFormat *> (formatter_);}

  };

#endif

