#include <string>
#include <algorithm>
#include <numeric>
#include <functional>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"

#include "SimCalorimetry/EcalEBTrigPrimAlgos/interface/EcalEBPhase2TrigPrimAlgo.h"

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame_Ph2.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTriggerElectronicsId.h"

#include "CondFormats/EcalObjects/interface/EcalTPGPedestals.h"
#include "CondFormats/DataRecord/interface/EcalTPGPedestalsRcd.h"

#include "DataFormats/EcalDigi/interface/EcalConstants.h"

#include <TTree.h>
#include <TMath.h>

//----------------------------------------------------------------------

const unsigned int EcalEBPhase2TrigPrimAlgo::nrSamples_ =
    ecalPh2::sampleSize;  // set to 16 samples, might change (less than 16) in the future
const unsigned int EcalEBPhase2TrigPrimAlgo::maxNrTowers_ = 2448;  // number of towers in EB

EcalEBPhase2TrigPrimAlgo::EcalEBPhase2TrigPrimAlgo(const EcalTrigTowerConstituentsMap *eTTmap,
                                                   const CaloGeometry *theGeometry,
                                                   int binofmax,
                                                   bool debug)
    : eTTmap_(eTTmap),
      theGeometry_(theGeometry),
      binOfMaximum_(binofmax),
      debug_(debug)

{
  maxNrSamples_ = ecalPh2::sampleSize;
  this->init();
}

void EcalEBPhase2TrigPrimAlgo::init() {
  theMapping_ = new EcalElectronicsMapping();
  // initialise data structures
  initStructures(towerMapEB_);
  hitTowers_.resize(maxNrTowers_);

  linearizer_ = new EcalEBPhase2Linearizer(debug_);
  lin_out_.resize(maxNrSamples_);

  amplitude_reconstructor_ = new EcalEBPhase2AmplitudeReconstructor(debug_);
  filt_out_.resize(maxNrSamples_);

  tpFormatter_ = new EcalEBPhase2TPFormatter(debug_);
  outEt_.resize(maxNrSamples_);
  outTime_.resize(maxNrSamples_);

  //

  time_reconstructor_ = new EcalEBPhase2TimeReconstructor(debug_);
  time_out_.resize(maxNrSamples_);
  spike_tagger_ = new EcalEBPhase2SpikeTagger(debug_);
}
//----------------------------------------------------------------------

EcalEBPhase2TrigPrimAlgo::~EcalEBPhase2TrigPrimAlgo() {
  delete linearizer_;
  delete amplitude_reconstructor_;
  delete time_reconstructor_;
  delete spike_tagger_;
  delete tpFormatter_;
  delete theMapping_;
}

void EcalEBPhase2TrigPrimAlgo::run(EBDigiCollectionPh2 const *digi, EcalEBPhase2TrigPrimDigiCollection &result) {
  if (debug_)
    LogDebug("") << "  EcalEBPhase2TrigPrimAlgo: digi size " << digi->size() << std::endl;

  EcalEBPhase2TriggerPrimitiveDigi tp;
  int firstSample = binOfMaximum_ - 1 - nrSamples_ / 2;
  int lastSample = binOfMaximum_ - 1 + nrSamples_ / 2;

  if (debug_) {
    LogDebug("") << "  binOfMaximum_ " << binOfMaximum_ << " nrSamples_" << nrSamples_ << std::endl;
    LogDebug("") << " first sample " << firstSample << " last " << lastSample << std::endl;
  }

  clean(towerMapEB_);
  fillMap(digi, towerMapEB_);

  int iChannel = 0;
  int nXinBCP = 0;
  for (int itow = 0; itow < nrTowers_; ++itow) {
    int index = hitTowers_[itow].first;
    const EcalTrigTowerDetId &thisTower = hitTowers_[itow].second;
    if (debug_)
      LogDebug("") << " Data for TOWER num " << itow << " index " << index << " TowerId " << thisTower << " zside "
                   << thisTower.zside() << " ieta " << thisTower.ieta() << " iphi " << thisTower.iphi() << " size "
                   << towerMapEB_[itow].size() << std::endl;

    // loop over all strips assigned to this trigger tower
    int nxstals = 0;
    for (unsigned int iStrip = 0; iStrip < towerMapEB_[itow].size(); ++iStrip) {
      if (debug_)
        LogDebug("") << " Data for STRIP num " << iStrip << std::endl;
      std::vector<EBDataFrame_Ph2> &dataFrames =
          (towerMapEB_[index])[iStrip].second;  //vector of dataframes for this strip, size; nr of crystals/strip

      nxstals = (towerMapEB_[index])[iStrip].first;
      if (nxstals <= 0)
        continue;
      if (debug_)
        LogDebug("") << " Number of xTals " << nxstals << std::endl;

      //const EcalTriggerElectronicsId elId = theMapping_->getTriggerElectronicsId(dataFrames[0].id());

      // loop over the xstals in a strip

      for (int iXstal = 0; iXstal < nxstals; iXstal++) {
        const EBDetId &myid = dataFrames[iXstal].id();

        nXinBCP++;
        if (debug_) {
          LogDebug("") << " Data for TOWER num " << itow << " index " << index << " TowerId " << thisTower << " size "
                       << towerMapEB_[itow].size() << std::endl;
          LogDebug("") << "nXinBCP " << nXinBCP << " myid rawId " << myid.rawId() << " xTal iEta " << myid.ieta()
                       << " iPhi " << myid.iphi() << std::endl;
        }

        tp = EcalEBPhase2TriggerPrimitiveDigi(myid);
        tp.setSize(nrSamples_);

        iChannel++;
        if (debug_) {
          LogDebug("") << " " << std::endl;
          LogDebug("") << " ******  iChannel " << iChannel << std::endl;
          for (int i = 0; i < dataFrames[iXstal].size(); i++) {
            LogDebug("") << " " << dataFrames[iXstal][i].adc();
          }
          LogDebug("") << " " << std::endl;
        }

        if (debug_) {
          LogDebug("") << std::endl;
          EBDetId id = dataFrames[iXstal].id();
          LogDebug("") << "iXstal= " << iXstal << std::endl;
          LogDebug("") << "iXstal= " << iXstal << " id " << id << " EcalDataFrame_Ph2 is: " << std::endl;
          for (int i = 0; i < dataFrames[iXstal].size(); i++) {
            LogDebug("") << " " << std::dec << dataFrames[iXstal][i].adc();
          }
          LogDebug("") << std::endl;
        }

        //   Call the linearizer
        this->getLinearizer()->setParameters(dataFrames[iXstal].id(), ecaltpPed_, ecaltpLin_, ecaltpgBadX_);
        this->getLinearizer()->process(dataFrames[iXstal], lin_out_);

        for (unsigned int i = 0; i < lin_out_.size(); i++) {
          if (lin_out_[i] > 0X3FFF)
            lin_out_[i] = 0X3FFF;
        }

        if (debug_) {
          LogDebug("") << "EcalEBPhase2TrigPrimAlgo output of linearize for channel " << iXstal << std::endl;
          for (unsigned int i = 0; i < lin_out_.size(); i++) {
            LogDebug("") << " " << std::dec << lin_out_[i];
          }
          LogDebug("") << std::endl;
        }

        // call spike finder right after the linearizer
        this->getSpikeTagger()->setParameters(dataFrames[iXstal].id(), ecaltpPed_, ecaltpLin_, ecaltpgBadX_);
        bool isASpike = this->getSpikeTagger()->process(lin_out_);

        //if (!isASpike) {

        // Call the amplitude reconstructor
        this->getAmplitudeFinder()->setParameters(myid.rawId(), ecaltpgAmplWeightMap_, ecaltpgWeightGroup_);
        this->getAmplitudeFinder()->process(lin_out_, filt_out_);

        if (debug_) {
          LogDebug("") << "EcalEBPhase2TrigPrimAlgo output of amp finder is a vector of size: " << std::dec
                       << time_out_.size() << std::endl;
          for (unsigned int ix = 0; ix < filt_out_.size(); ix++) {
            LogDebug("") << std::dec << filt_out_[ix] << " ";
          }
          LogDebug("") << std::endl;
        }

        if (debug_) {
          LogDebug("") << " Ampl "
                       << " ";
          for (unsigned int ix = 0; ix < filt_out_.size(); ix++) {
            LogDebug("") << std::dec << filt_out_[ix] << " ";
          }
          LogDebug("") << std::endl;
        }

        // call time finder
        this->getTimeFinder()->setParameters(myid.rawId(), ecaltpgTimeWeightMap_, ecaltpgWeightGroup_);
        this->getTimeFinder()->process(lin_out_, filt_out_, time_out_);

        if (debug_) {
          LogDebug("") << " Time "
                       << " ";
          for (unsigned int ix = 0; ix < time_out_.size(); ix++) {
            LogDebug("") << std::dec << time_out_[ix] << " ";
          }
          LogDebug("") << std::endl;
        }

        if (debug_) {
          LogDebug("") << "EcalEBPhase2TrigPrimAlgo output of timefinder is a vector of size: " << std::dec
                       << time_out_.size() << std::endl;
          for (unsigned int ix = 0; ix < time_out_.size(); ix++) {
            LogDebug("") << std::dec << time_out_[ix] << " ";
          }
          LogDebug("") << std::endl;
        }

        this->getTPFormatter()->process(filt_out_, time_out_, outEt_, outTime_);

        if (debug_) {
          LogDebug("") << " compressed Et "
                       << " ";
          for (unsigned int iSample = 0; iSample < outEt_.size(); ++iSample) {
            LogDebug("") << outEt_[iSample] << " ";
          }
          LogDebug("") << std::endl;

          LogDebug("") << " compressed time "
                       << " ";
          for (unsigned int iSample = 0; iSample < outEt_.size(); ++iSample) {
            LogDebug("") << outTime_[iSample] << " ";
          }
          LogDebug("") << std::endl;
        }

        if (debug_) {
          LogDebug("") << " EcalEBPhase2TrigPrimAlgo  after getting the formatter " << std::endl;
          for (unsigned int iSample = 0; iSample < outEt_.size(); ++iSample) {
            LogDebug("") << " outEt " << outEt_[iSample] << " outTime " << outTime_[iSample] << " ";
          }
          LogDebug("") << std::endl;
        }

        // } not a spike

        // create the final TP samples
        int etInADC = 0;
        ;
        int64_t time = -999;
        int nSam = 0;
        for (int iSample = 0; iSample < 16; ++iSample) {
          etInADC = outEt_[iSample];
          time = outTime_[iSample];
          if (debug_) {
            LogDebug("") << "TrigPrimAlgo   outEt " << outEt_[iSample] << " outTime " << outTime_[iSample] << std::endl;
            LogDebug("") << "TrigPrimAlgo etInADCt " << outEt_[iSample] << " outTime " << time << std::endl;
          }

          tp.setSample(nSam, EcalEBPhase2TriggerPrimitiveSample(etInADC, isASpike, time));
          nSam++;
        }

        result.push_back(tp);

      }  // Loop over the xStals

    }  //loop over strips in one tower

    if (debug_) {
      if (nXinBCP > 0)
        LogDebug("") << " Accepted xTals " << nXinBCP << std::endl;
    }
  }
}

//----------------------------------------------------------------------

int EcalEBPhase2TrigPrimAlgo::findStripNr(const EBDetId &id) {
  int stripnr;
  int n = ((id.ic() - 1) % 100) / 20;  //20 corresponds to 4 * ecal_barrel_crystals_per_strip FIXME!!
  if (id.ieta() < 0)
    stripnr = n + 1;
  else
    stripnr = nbMaxStrips_ - n;
  return stripnr;
}
