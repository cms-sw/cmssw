#include <iostream>
#include <cmath>

#include "SimTracker/SiPhase2Digitizer/plugins/PixelBrickedDigitizerAlgorithm.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationOfflineSimService.h"

#include "CondFormats/SiPixelObjects/interface/GlobalPixel.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelLorentzAngleSimRcd.h"

// Geometry
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

using namespace edm;
using namespace sipixelobjects;

PixelBrickedDigitizerAlgorithm::PixelBrickedDigitizerAlgorithm(const edm::ParameterSet& conf, edm::ConsumesCollector iC)
    : PixelDigitizerAlgorithm(conf, iC)
/*    odd_row_interchannelCoupling_next_row_(conf.getParameter<ParameterSet>("PixelBrickedDigitizerAlgorithm")
                                                 .getParameter<double>("Odd_row_interchannelCoupling_next_row"))
         even_row_interchannelCoupling_next_row_(conf.getParameter<ParameterSet>("PixelBrickedDigitizerAlgorithm")
                                                  .getParameter<double>("Even_row_interchannelCoupling_next_row")),
      odd_column_interchannelCoupling_next_column_(
          conf.getParameter<ParameterSet>("PixelBrickedDigitizerAlgorithm")
              .getParameter<double>("Odd_column_interchannelCoupling_next_column")),
      even_column_interchannelCoupling_next_column_(
          conf.getParameter<ParameterSet>("PixelBrickedDigitizerAlgorithm")
	  .getParameter<double>("Even_column_interchannelCoupling_next_column"))*/
{
  even_row_interchannelCoupling_next_row_ = conf.getParameter<ParameterSet>("PixelBrickedDigitizerAlgorithm")
                                                .getParameter<double>("Even_row_interchannelCoupling_next_row");
  pixelFlag_ = true;
  LogDebug("PixelBrickedDigitizerAlgorithm")
      << "Algorithm constructed "
      << "Configuration parameters:"
      << "Threshold/Gain = "
      << "threshold in electron Endcap = " << theThresholdInE_Endcap_
      << "threshold in electron Barrel = " << theThresholdInE_Barrel_ << " " << theElectronPerADC_ << " "
      << theAdcFullScale_ << " The delta cut-off is set to " << tMax_ << " pix-inefficiency " << addPixelInefficiency_;
}
PixelBrickedDigitizerAlgorithm::~PixelBrickedDigitizerAlgorithm() {
  LogDebug("PixelBrickedDigitizerAlgorithm") << "Algorithm deleted";
}
void PixelBrickedDigitizerAlgorithm::induce_signal(std::vector<PSimHit>::const_iterator inputBegin,
                                                   const PSimHit& hit,
                                                   const size_t hitIndex,
                                                   const size_t firstHitIndex,
                                                   const uint32_t tofBin,
                                                   const Phase2TrackerGeomDetUnit* pixdet,
                                                   const std::vector<digitizerUtility::SignalPoint>& collection_points) {
  // X  - Rows, Left-Right, 160, (1.6cm)   for barrel
  // Y  - Columns, Down-Up, 416, (6.4cm)
  const Phase2TrackerTopology* topol = &pixdet->specificTopology();
  uint32_t detID = pixdet->geographicalId().rawId();
  signal_map_type& theSignal = _signal[detID];

  // local map to store pixels hit by 1 Hit.
  using hit_map_type = std::map<int, float, std::less<int> >;
  hit_map_type hit_signal;

  // Assign signals to readout channels and store sorted by channel number
  // Iterate over collection points on the collection plane
  for (auto const& v : collection_points) {
    float CloudCenterX = v.position().x();  // Charge position in x
    float CloudCenterY = v.position().y();  //                 in y
    float SigmaX = v.sigma_x();             // Charge spread in x
    float SigmaY = v.sigma_y();             //               in y
    float Charge = v.amplitude();           // Charge amplitude

    // Find the maximum cloud spread in 2D plane , assume 3*sigma
    float CloudRight = CloudCenterX + clusterWidth_ * SigmaX;
    float CloudLeft = CloudCenterX - clusterWidth_ * SigmaX;
    float CloudUp = CloudCenterY + clusterWidth_ * SigmaY;
    float CloudDown = CloudCenterY - clusterWidth_ * SigmaY;

    // Define 2D cloud limit points
    LocalPoint PointRightUp = LocalPoint(CloudRight, CloudUp);
    LocalPoint PointLeftDown = LocalPoint(CloudLeft, CloudDown);

    // This points can be located outside the sensor area.
    // The conversion to measurement point does not check for that
    // so the returned pixel index might be wrong (outside range).
    // We rely on the limits check below to fix this.
    // But remember whatever we do here THE CHARGE OUTSIDE THE ACTIVE
    // PIXEL AREA IS LOST, it should not be collected.

    // Convert the 2D points to pixel indices
    MeasurementPoint mp = topol->measurementPosition(PointRightUp);
    //MeasurementPoint mp_bricked = topol->measurementPosition(PointRightUp);
    int IPixRightUpX = static_cast<int>(std::floor(mp.x()));  // cast reqd.
    //int IPixRightUpY = static_cast<int>(std::floor(mp.y()));

    int numColumns = topol->ncolumns();  // det module number of cols&rows
    int numRows = topol->nrows();
    IPixRightUpX = numRows > IPixRightUpX ? IPixRightUpX : numRows - 1;

    //Specific to bricked geometry
    int IPixRightUpY = static_cast<int>(mp.y() - 0.5 * (IPixRightUpX % 2));

    mp = topol->measurementPosition(PointLeftDown);

    int IPixLeftDownX = static_cast<int>(std::floor(mp.x()));

    IPixLeftDownX = 0 < IPixLeftDownX ? IPixLeftDownX : 0;

    //Specific to bricked geometry
    int IPixLeftDownY = static_cast<int>(mp.y() - 0.5 * (IPixLeftDownX % 2));  //changed in case negative value

    IPixRightUpY = numColumns > IPixRightUpY ? IPixRightUpY : numColumns - 1;
    IPixLeftDownY = 0 < IPixLeftDownY ? IPixLeftDownY : 0;

    // First integrate charge strips in x
    hit_map_type x;
    for (int ix = IPixLeftDownX; ix <= IPixRightUpX; ++ix) {  // loop over x index
      float xLB, LowerBound;
      // Why is set to 0 if ix=0, does it meen that we accept charge
      // outside the sensor?
      if (ix == 0 || SigmaX == 0.) {  // skip for surface segemnts
        LowerBound = 0.;
      } else {
        mp = MeasurementPoint(ix, 0.0);
        xLB = topol->localPosition(mp).x();
        LowerBound = 1 - calcQ((xLB - CloudCenterX) / SigmaX);
      }

      float xUB, UpperBound;
      if (ix == numRows - 1 || SigmaX == 0.) {
        UpperBound = 1.;
      } else {
        mp = MeasurementPoint(ix + 1, 0.0);
        xUB = topol->localPosition(mp).x();
        UpperBound = 1. - calcQ((xUB - CloudCenterX) / SigmaX);
      }
      float TotalIntegrationRange = UpperBound - LowerBound;  // get strip
      x.emplace(ix, TotalIntegrationRange);                   // save strip integral
    }

    // Now integrate strips in y. Two maps will be filled: y and y_bricked which will both be used for the induced signal.
    int IPixLeftDownY_bricked = IPixLeftDownY;
    int IPixRightUpY_bricked = IPixRightUpY;

    //Specific to bricked geometry
    IPixRightUpY = std::min(IPixRightUpY + int((IPixRightUpX % 2)), numColumns - 1);

    //This map will be twice as large as the non-bricked hit map in y to harbor both the integrated charge from the bricked and non-bricked columns.
    hit_map_type y;
    for (int iy = IPixLeftDownY; iy <= IPixRightUpY; ++iy) {  // loop over y index
      float yLB, LowerBound;
      if (iy == 0 || SigmaY == 0.) {
        LowerBound = 0.;
      } else {
        mp = MeasurementPoint(0.0, iy);
        yLB = topol->localPosition(mp).y();
        LowerBound = 1. - calcQ((yLB - CloudCenterY) / SigmaY);
      }

      float yUB, UpperBound;
      if (iy >= numColumns - 1 || SigmaY == 0.) {
        UpperBound = 1.;
      } else {
        mp = MeasurementPoint(0.0, iy + 1);
        yUB = topol->localPosition(mp).y();
        UpperBound = 1. - calcQ((yUB - CloudCenterY) / SigmaY);
      }

      float TotalIntegrationRange = UpperBound - LowerBound;

      //Even indices correspond to the non-bricked columns
      y.emplace(2 * iy, TotalIntegrationRange);  // save strip integral
    }

    IPixLeftDownY_bricked = std::max(IPixLeftDownY_bricked - int((!(IPixLeftDownX % 2))), 0);
    for (int iy = IPixLeftDownY_bricked; iy <= IPixRightUpY_bricked; ++iy) {  // loop over y index
      float yLB, LowerBound;
      if (iy == 0 || SigmaY == 0.) {
        LowerBound = 0.;
      } else {
        mp = MeasurementPoint(0.0, iy + 0.5);
        yLB = topol->localPosition(mp).y();
        LowerBound = 1. - calcQ((yLB - CloudCenterY) / SigmaY);
      }

      float yUB, UpperBound;
      if (iy >= numColumns || SigmaY == 0.) {  // This was changed for bricked pixels
        UpperBound = 1.;
      } else {
        mp = MeasurementPoint(0.0, iy + 1.5);
        yUB = topol->localPosition(mp).y();
        UpperBound = 1. - calcQ((yUB - CloudCenterY) / SigmaY);
      }

      float TotalIntegrationRange = UpperBound - LowerBound;
      //Odd indices correspond to bricked columns
      y.emplace(2 * iy + 1, TotalIntegrationRange);  // save strip integral
    }                                                //loop over y index

    // Get the 2D charge integrals by folding x and y strips
    for (int ix = IPixLeftDownX; ix <= IPixRightUpX; ++ix) {                                 // loop over x index
      for (int iy = std::max(0, IPixLeftDownY - int((ix % 2))); iy <= IPixRightUpY; ++iy) {  // loop over y index
        int iy_considered = iy * 2 + ix % 2;
        float ChargeFraction = Charge * x[ix] * y[iy_considered];

        int chanFired = -1;
        if (ChargeFraction > 0.) {
          chanFired =
              pixelFlag_ ? PixelDigi::pixelToChannel(ix, iy) : Phase2TrackerDigi::pixelToChannel(ix, iy);  // Get index
          // Load the ampl
          hit_signal[chanFired] += ChargeFraction;
        }
      }
    }  //x loop
  }    //collection loop
  // Fill the global map with all hit pixels from this event
  float corr_time = hit.tof() - pixdet->surface().toGlobal(hit.localPosition()).mag() * c_inv;
  for (auto const& hit_s : hit_signal) {
    int chan = hit_s.first;
    theSignal[chan] += (makeDigiSimLinks_ ? digitizerUtility::Ph2Amplitude(
                                                hit_s.second, &hit, hit_s.second, corr_time, hitIndex, tofBin)
                                          : digitizerUtility::Ph2Amplitude(hit_s.second, nullptr, hit_s.second));
  }
}
