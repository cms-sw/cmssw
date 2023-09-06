#include <string>
#include <algorithm>
#include <numeric>
#include <functional>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
//#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"

#include "SimCalorimetry/EcalEBTrigPrimAlgos/interface/EcalEBPhase2TrigPrimAlgo.h"

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame_Ph2.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTriggerElectronicsId.h"

#include "CondFormats/EcalObjects/interface/EcalTPGPedestals.h"
#include "CondFormats/DataRecord/interface/EcalTPGPedestalsRcd.h"

#include "DataFormats/EcalDigi/interface/EcalConstants.h"

#include <TTree.h>
#include <TMath.h>

//----------------------------------------------------------------------

const unsigned int EcalEBPhase2TrigPrimAlgo::nrSamples_ = 16;
const unsigned int EcalEBPhase2TrigPrimAlgo::maxNrTowers_ = 2448;
const unsigned int EcalEBPhase2TrigPrimAlgo::maxNrSamplesOut_ = ecalPh2::sampleSize;

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
}

void EcalEBPhase2TrigPrimAlgo::run(EBDigiCollectionPh2 const *digi, EcalEBTrigPrimDigiCollection &result) {
  //typedef typename Coll::Digi Digi;
  if (debug_) {
    std::cout << "  EcalEBPhase2TrigPrimAlgo: Testing that the algorythm with digis is well plugged " << std::endl;
    std::cout << "  EcalEBPhase2TrigPrimAlgo: digi size " << digi->size() << std::endl;
  }

  EcalEBTriggerPrimitiveDigi tp;
  int firstSample = binOfMaximum_ - 1 - nrSamples_ / 2;
  int lastSample = binOfMaximum_ - 1 + nrSamples_ / 2;

  if (debug_) {
    std::cout << "  binOfMaximum_ " << binOfMaximum_ << " nrSamples_" << nrSamples_ << std::endl;
    std::cout << " first sample " << firstSample << " last " << lastSample << std::endl;
  }

  clean(towerMapEB_);
  fillMap(digi, towerMapEB_);

  int iChannel = 0;
  int nXinBCP = 0;
  for (int itow = 0; itow < nrTowers_; ++itow) {
    int index = hitTowers_[itow].first;
    const EcalTrigTowerDetId &thisTower = hitTowers_[itow].second;
    if (debug_)
      std::cout << " Data for TOWER num " << itow << " index " << index << " TowerId " << thisTower << " zside "
                << thisTower.zside() << " ieta " << thisTower.ieta() << " iphi " << thisTower.iphi() << " size "
                << towerMapEB_[itow].size() << std::endl;

    // loop over all strips assigned to this trigger tower
    int nxstals = 0;
    for (unsigned int iStrip = 0; iStrip < towerMapEB_[itow].size(); ++iStrip) {
      if (debug_)
        std::cout << " Data for STRIP num " << iStrip << std::endl;
      std::vector<EBDataFrame_Ph2> &dataFrames =
          (towerMapEB_[index])[iStrip].second;  //vector of dataframes for this strip, size; nr of crystals/strip

      nxstals = (towerMapEB_[index])[iStrip].first;
      if (nxstals <= 0)
        continue;
      if (debug_)
        std::cout << " Number of xTals " << nxstals << std::endl;

      //const EcalTriggerElectronicsId elId = theMapping_->getTriggerElectronicsId(dataFrames[0].id());

      // loop over the xstals in a strip

      for (int iXstal = 0; iXstal < nxstals; iXstal++) {
        const EBDetId &myid = dataFrames[iXstal].id();

        nXinBCP++;
        if (debug_) {
          std::cout << " Data for TOWER num " << itow << " index " << index << " TowerId " << thisTower << " size "
                    << towerMapEB_[itow].size() << std::endl;
          std::cout << "nXinBCP " << nXinBCP << " myid rawId " << myid.rawId() << " xTal iEta " << myid.ieta()
                    << " iPhi " << myid.iphi() << std::endl;
        }

        tp = EcalEBTriggerPrimitiveDigi(myid);
        tp.setSize(nrSamples_);

        iChannel++;
        if (debug_) {
          std::cout << " " << std::endl;
          std::cout << " ******  iChannel " << iChannel << std::endl;
          for (int i = 0; i < dataFrames[iXstal].size(); i++) {
            std::cout << " " << dataFrames[iXstal][i].adc();
          }
          std::cout << " " << std::endl;
        }

        if (debug_) {
          std::cout << std::endl;
          EBDetId id = dataFrames[iXstal].id();
          std::cout << "iXstal= " << iXstal << std::endl;
          std::cout << "iXstal= " << iXstal << " id " << id << " EcalDataFrame_Ph2 is: " << std::endl;
          for (int i = 0; i < dataFrames[iXstal].size(); i++) {
            std::cout << " " << std::dec << dataFrames[iXstal][i].adc();
          }
          std::cout << std::endl;
        }

        //   Call the linearizer
        this->getLinearizer()->setParameters(dataFrames[iXstal].id(), ecaltpPed_, ecaltpLin_, ecaltpgBadX_);
        this->getLinearizer()->process(dataFrames[iXstal], lin_out_);

        for (unsigned int i = 0; i < lin_out_.size(); i++) {
          if (lin_out_[i] > 0X3FFF)
            lin_out_[i] = 0X3FFF;
        }

        if (debug_) {
          std::cout << "EcalEBPhase2TrigPrimAlgo output of linearize for channel " << iXstal << std::endl;
          for (unsigned int i = 0; i < lin_out_.size(); i++) {
            std::cout << " " << std::dec << lin_out_[i];
          }
          std::cout << std::endl;
        }

        // call spike finder right after the linearizer
        this->getSpikeTagger()->setParameters(dataFrames[iXstal].id(), ecaltpPed_, ecaltpLin_, ecaltpgBadX_);
        bool isASpike = this->getSpikeTagger()->process(lin_out_);

        //if (!isASpike) {

        // Call the amplitude reconstructor
        this->getAmplitudeFinder()->setParameters(myid.rawId(), ecaltpgAmplWeightMap_, ecaltpgWeightGroup_);
        this->getAmplitudeFinder()->process(lin_out_, filt_out_);

        if (debug_) {
          std::cout << "EcalEBPhase2TrigPrimAlgo output of amp finder is a vector of size: " << std::dec
                    << time_out_.size() << std::endl;
          for (unsigned int ix = 0; ix < filt_out_.size(); ix++) {
            std::cout << std::dec << filt_out_[ix] << " ";
          }
          std::cout << std::endl;
        }

        if (debug_) {
          std::cout << " Ampl "
                    << " ";
          for (unsigned int ix = 0; ix < filt_out_.size(); ix++) {
            std::cout << std::dec << filt_out_[ix] << " ";
          }
          std::cout << std::endl;
        }

        // call time finder
        this->getTimeFinder()->setParameters(myid.rawId(), ecaltpgTimeWeightMap_, ecaltpgWeightGroup_);
        this->getTimeFinder()->process(lin_out_, filt_out_, time_out_);

        if (debug_) {
          std::cout << " Time "
                    << " ";
          for (unsigned int ix = 0; ix < time_out_.size(); ix++) {
            std::cout << std::dec << time_out_[ix] << " ";
          }
          std::cout << std::endl;
        }

        if (debug_) {
          std::cout << "EcalEBPhase2TrigPrimAlgo output of timefinder is a vector of size: " << std::dec
                    << time_out_.size() << std::endl;
          for (unsigned int ix = 0; ix < time_out_.size(); ix++) {
            std::cout << std::dec << time_out_[ix] << " ";
          }
          std::cout << std::endl;
        }

        this->getTPFormatter()->process(filt_out_, time_out_, outEt_, outTime_);

        if (debug_) {
          std::cout << " compressed Et "
                    << " ";
          for (unsigned int iSample = 0; iSample < outEt_.size(); ++iSample) {
            std::cout << outEt_[iSample] << " ";
          }
          std::cout << std::endl;

          std::cout << " compressed time "
                    << " ";
          for (unsigned int iSample = 0; iSample < outEt_.size(); ++iSample) {
            std::cout << outTime_[iSample] << " ";
          }
          std::cout << std::endl;
        }

        if (debug_) {
          std::cout << " EcalEBPhase2TrigPrimAlgo  after getting the formatter " << std::endl;
          for (unsigned int iSample = 0; iSample < outEt_.size(); ++iSample) {
            std::cout << " outEt " << outEt_[iSample] << " outTime " << outTime_[iSample] << " ";
          }
          std::cout << std::endl;
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
            std::cout << "TrigPrimAlgo   outEt " << outEt_[iSample] << " outTime " << outTime_[iSample] << std::endl;
            std::cout << "TrigPrimAlgo etInADCt " << outEt_[iSample] << " outTime " << time << std::endl;
          }

          tp.setSample(nSam, EcalEBTriggerPrimitiveSample(etInADC, isASpike, time));
          nSam++;
        }

        result.push_back(tp);

      }  // Loop over the xStals

    }  //loop over strips in one tower

    if (debug_) {
      if (nXinBCP > 0)
        std::cout << " Accepted xTals " << nXinBCP << std::endl;
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
