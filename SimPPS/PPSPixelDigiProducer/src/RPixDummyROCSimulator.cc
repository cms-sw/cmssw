#include "SimPPS/PPSPixelDigiProducer/interface/RPixDummyROCSimulator.h"
#include <vector>
#include "TRandom.h"
#include <iostream>

RPixDummyROCSimulator::RPixDummyROCSimulator(const edm::ParameterSet &params, uint32_t det_id) : det_id_(det_id) {
  threshold_ = params.getParameter<double>("RPixDummyROCThreshold");
  electron_per_adc_ = params.getParameter<double>("RPixDummyROCElectronPerADC");
  VcaltoElectronGain_ = params.getParameter<int>("VCaltoElectronGain");
  VcaltoElectronOffset_ = params.getParameter<int>("VCaltoElectronOffset");
  doSingleCalibration_ = params.getParameter<bool>("doSingleCalibration");
  dead_pixel_probability_ = params.getParameter<double>("RPixDeadPixelProbability");
  dead_pixels_simulation_on_ = params.getParameter<bool>("RPixDeadPixelSimulationOn");
  verbosity_ = params.getParameter<int>("RPixVerbosity");
  links_persistence_ = params.getParameter<bool>("CTPPSPixelDigiSimHitRelationsPersistence");
}

void RPixDummyROCSimulator::ConvertChargeToHits(
    const std::map<unsigned short, double> &signals,
    std::map<unsigned short, std::vector<std::pair<int, double> > > &theSignalProvenance,
    std::vector<CTPPSPixelDigi> &output_digi,
    std::vector<std::vector<std::pair<int, double> > > &output_digi_links,
    const CTPPSPixelGainCalibrations *pcalibrations) {
  for (std::map<unsigned short, double>::const_iterator i = signals.begin(); i != signals.end(); ++i) {
    //one threshold per hybrid
    unsigned short pixel_no = i->first;
    if (verbosity_)
      edm::LogInfo("PPS") << "RPixDummyROCSimulator "
                          << "Dummy ROC adc and threshold : " << i->second << ", " << threshold_;
    if (i->second > threshold_ && (!dead_pixels_simulation_on_ || dead_pixels_.find(pixel_no) == dead_pixels_.end())) {
      float gain = 0;
      float pedestal = 0;
      int adc = 0;
      uint32_t col = pixel_no / 160;
      uint32_t row = pixel_no % 160;

      const CTPPSPixelGainCalibration &DetCalibs = pcalibrations->getGainCalibration(det_id_);

      // Avoid exception due to col > 103 in case of 2x2 plane. To be removed
      if (col >= DetCalibs.getNCols())
        continue;

      if (doSingleCalibration_) {
        adc = int(round(i->second / electron_per_adc_));
      } else {
        if (DetCalibs.getDetId() != 0) {
          gain = DetCalibs.getGain(col, row) * highRangeCal_ / lowRangeCal_;  // *highRangeCal/lowRangeCal
          pedestal = DetCalibs.getPed(col, row);
          adc = int(round((i->second - VcaltoElectronOffset_) / (gain * VcaltoElectronGain_) + pedestal));
        }
      }
      /// set maximum for 8 bits adc
      if (adc >= maxADC_)
        adc = maxADC_;
      if (adc < 0)
        adc = 0;
      output_digi.push_back(CTPPSPixelDigi(row, col, adc));
      if (links_persistence_) {
        output_digi_links.push_back(theSignalProvenance[pixel_no]);
        if (verbosity_) {
          edm::LogInfo("PPS") << "RPixDummyROCSimulator "
                              << "digi links size=" << theSignalProvenance[pixel_no].size();
          for (unsigned int u = 0; u < theSignalProvenance[pixel_no].size(); ++u) {
            edm::LogInfo("PPS") << "RPixDummyROCSimulator "
                                << "   digi: particle=" << theSignalProvenance[pixel_no][u].first
                                << " energy [electrons]=" << theSignalProvenance[pixel_no][u].second;
          }
        }
      }
    }
  }

  if (verbosity_) {
    for (unsigned int i = 0; i < output_digi.size(); ++i) {
      edm::LogInfo("RPixDummyROCSimulator")
          << "Dummy ROC Simulator " << det_id_ << "     row= "  //output_digi[i].GetDetId()<<" "
          << output_digi[i].row() << "   col= " << output_digi[i].column() << "   adc= " << output_digi[i].adc();
    }
  }
}
