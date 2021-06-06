#include "SimPPS/PPSPixelDigiProducer/interface/RPixChargeShare.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include <iostream>
#include <fstream>

RPixChargeShare::RPixChargeShare(const edm::ParameterSet &params, uint32_t det_id, const PPSPixelTopology &ppt)
    : det_id_(det_id) {
  verbosity_ = params.getParameter<int>("RPixVerbosity");
  signalCoupling_.clear();
  ChargeMapFile2E_[0] = params.getParameter<std::string>("ChargeMapFile2E");
  ChargeMapFile2E_[1] = params.getParameter<std::string>("ChargeMapFile2E_2X");
  ChargeMapFile2E_[2] = params.getParameter<std::string>("ChargeMapFile2E_2Y");
  ChargeMapFile2E_[3] = params.getParameter<std::string>("ChargeMapFile2E_2X2Y");

  double coupling_constant_ = params.getParameter<double>("RPixCoupling");

  signalCoupling_.push_back(coupling_constant_);
  signalCoupling_.push_back((1.0 - coupling_constant_) / 2);

  no_of_pixels_ = ppt.getNoPixels();

  double xMap, yMap;
  double chargeprobcollect;
  int xUpper[] = {75, 150, 75, 150};
  int yUpper[] = {50, 50, 100, 100};
  int ix, iy;
  for (int i = 0; i < 4; i++) {
    edm::FileInPath filename(ChargeMapFile2E_[i]);
    std::ifstream fChargeMap(filename.fullPath().c_str());
    if (fChargeMap.is_open()) {
      while (fChargeMap >> xMap >> yMap >> chargeprobcollect) {
        ix = int((xMap + xUpper[i]) / 5);
        iy = int((yMap + yUpper[i]) / 5);
        chargeMap2E_[i][ix][iy] = chargeprobcollect;
      }
      fChargeMap.close();
    } else
      throw cms::Exception("PPS RPixChargeShare") << "Charge map file not found";
  }
}

std::map<unsigned short, double> RPixChargeShare::Share(const std::vector<RPixSignalPoint> &charge_map,
                                                        const PPSPixelTopology &ppt) {
  std::map<unsigned short, double> thePixelChargeMap;
  if (verbosity_ > 1)
    edm::LogInfo("PPS") << "RPixChargeShare " << det_id_ << " : Clouds to be induced= " << charge_map.size();

  double cH = 0;

  for (std::vector<RPixSignalPoint>::const_iterator i = charge_map.begin(); i != charge_map.end(); ++i) {
    double hit_pos_x, hit_pos_y;
    // Used to avoid the abort due to hits out of detector
    if (((*i).Position().x() + ppt.getSimXWidth() / 2.) < 0 ||
        ((*i).Position().x() + ppt.getSimXWidth() / 2.) > ppt.getSimXWidth()) {
      edm::LogInfo("PPS RPixChargeShare")
          << "**** Attention ((*i).Position().x()+simX_width_/2.)<0||((*i).Position().x()+simX_width_/2.)>simX_width  ";
      edm::LogInfo("PPS RPixChargeShare") << "(*i).Position().x() = " << (*i).Position().x();
      continue;
    }
    if (((*i).Position().y() + ppt.getSimYWidth() / 2.) < 0 ||
        ((*i).Position().y() + ppt.getSimYWidth() / 2.) > ppt.getSimYWidth()) {
      edm::LogInfo("PPS RPixChargeShare")
          << "**** Attention ((*i).Position().y()+simY_width_/2.)<0||((*i).Position().y()+simY_width_/2.)>simY_width  ";
      edm::LogInfo("PPS RPixChargeShare") << "(*i).Position().y() = " << (*i).Position().y();
      continue;
    }

    PPSPixelTopology::PixelInfo relevant_pixels =
        ppt.getPixelsInvolved((*i).Position().x(), (*i).Position().y(), (*i).Sigma(), hit_pos_x, hit_pos_y);
    double effic = relevant_pixels.effFactor();
    if (effic < 1.e-4)
      continue;

    unsigned short pixel_no = ppt.pixelIndex(relevant_pixels);

    double charge_in_pixel = (*i).Charge() * effic;

    cH += charge_in_pixel;

    if (verbosity_ > 1)
      edm::LogInfo("PPS RPixChargeShare ") << "Efficiency in detector " << det_id_ << " and pixel no " << pixel_no
                                           << "  : " << effic << "  ch: " << charge_in_pixel << "   CHtot: " << cH;

    if (signalCoupling_[0] == 0.) {
      thePixelChargeMap[pixel_no] += charge_in_pixel;
    } else {
      int pixel_row = relevant_pixels.pixelRowNo();
      int pixel_col = relevant_pixels.pixelColNo();
      double pixel_lower_x = 0;
      double pixel_lower_y = 0;
      double pixel_upper_x = 0;
      double pixel_upper_y = 0;
      int psize = 0;
      ppt.pixelRange(pixel_row, pixel_col, pixel_lower_x, pixel_upper_x, pixel_lower_y, pixel_upper_y);
      double pixel_width_x = pixel_upper_x - pixel_lower_x;
      double pixel_width_y = pixel_upper_y - pixel_lower_y;
      double pixel_center_x = pixel_lower_x + (pixel_width_x) / 2.;
      double pixel_center_y = pixel_lower_y + (pixel_width_y) / 2.;

      /*
  First/last rows/columns are bigger due to sensor edges. 0.2mm is added to the pixel pitch but 
  only 0.15mm is considered sensitive. 
The pixel size is standard (100x150) but the actual one is +200 and the sensitive +150.
  Row 0: 100x150 -> 250x150 sensitive (lower x 50, higher x 300) but charge share only in 100x150 and
  x center in lower_x + active_edge + pitch_x/2.
  Row 159: 100x150 -> 250x150 sensitive (lower x 16300, higher x 16550) but charge share only in 100x150 and
  x center in lower_x + pitch_x/2.
*/
      if (pixel_row == 0) {
        pixel_width_x = ppt.getPitchSimX();
        pixel_center_x = pixel_lower_x + ppt.getPhysActiveEdgeDist() + ppt.getPitchSimX() / 2.;
        if (fabs((*i).Position().x() - pixel_center_x) > ppt.getPitchSimX() / 2.) {
          thePixelChargeMap[pixel_no] += charge_in_pixel;
          continue;
        }
        if (pixel_col == 0) {
          pixel_width_y = ppt.getPitchSimY();
          pixel_center_y = pixel_lower_y + ppt.getPhysActiveEdgeDist() + ppt.getPitchSimY() / 2.;
          if (fabs((*i).Position().y() - pixel_center_y) > ppt.getPitchSimY() / 2.) {
            thePixelChargeMap[pixel_no] += charge_in_pixel;
            continue;
          }
        }
        if (pixel_col == ppt.getNoPixelsSimY() - 1) {
          pixel_width_y = ppt.getPitchSimY();
          pixel_center_y = pixel_lower_y + ppt.getPitchSimY() / 2.;
          if (fabs((*i).Position().y() - pixel_center_y) > ppt.getPitchSimY() / 2.) {
            thePixelChargeMap[pixel_no] += charge_in_pixel;
            continue;
          }
        }
      }
      if (pixel_row == ppt.getNoPixelsSimX() - 1) {
        pixel_width_x = ppt.getPitchSimX();
        pixel_center_x = pixel_lower_x + ppt.getPitchSimX() / 2.;
        if (fabs((*i).Position().x() - pixel_center_x) > ppt.getPitchSimX() / 2.) {
          thePixelChargeMap[pixel_no] += charge_in_pixel;
          continue;
        }
        if (pixel_col == 0) {
          pixel_width_y = ppt.getPitchSimY();
          pixel_center_y = pixel_lower_y + ppt.getPhysActiveEdgeDist() + ppt.getPitchSimY() / 2.;
          if (fabs((*i).Position().y() - pixel_center_y) > ppt.getPitchSimY() / 2.) {
            thePixelChargeMap[pixel_no] += charge_in_pixel;
            continue;
          }
        }
        if (pixel_col == ppt.getNoPixelsSimY() - 1) {
          pixel_width_y = ppt.getPitchSimY();
          pixel_center_y = pixel_lower_y + ppt.getPitchSimY() / 2.;
          if (fabs((*i).Position().y() - pixel_center_y) > ppt.getPitchSimY() / 2.) {
            thePixelChargeMap[pixel_no] += charge_in_pixel;
            continue;
          }
        }
      }
      if (pixel_col == 0) {
        pixel_width_y = ppt.getPitchSimY();
        pixel_center_y = pixel_lower_y + ppt.getPhysActiveEdgeDist() + ppt.getPitchSimY() / 2.;
        if (fabs((*i).Position().y() - pixel_center_y) > ppt.getPitchSimY() / 2.) {
          thePixelChargeMap[pixel_no] += charge_in_pixel;
          continue;
        }
      }
      if (pixel_col == ppt.getNoPixelsSimY() - 1) {
        pixel_width_y = ppt.getPitchSimY();
        pixel_center_y = pixel_lower_y + ppt.getPitchSimY() / 2.;
        if (fabs((*i).Position().y() - pixel_center_y) > ppt.getPitchSimY() / 2.) {
          thePixelChargeMap[pixel_no] += charge_in_pixel;
          continue;
        }
      }

      // xbin and ybin are coordinates (um) inside the pixel as in the test beam, swapped wrt plane coordinates.
      int xbin = int((((*i).Position().y() - pixel_center_y) + pixel_width_y / 2.) * 1.e3 / 5.);
      int ybin = int((((*i).Position().x() - pixel_center_x) + pixel_width_x / 2.) * 1.e3 / 5.);
      if (pixel_width_x < 0.11 && pixel_width_y < 0.151) {  // pixel 100x150 um^2
        psize = 0;
        if (xbin > xBinMax_[psize] || ybin > yBinMax_[psize]) {
          edm::LogError("PPS RPixChargeShare") << " Array index out of bounds 0";
          continue;
        }
      }
      if (pixel_width_x > 0.11 && pixel_width_y < 0.151) {  // pixel 200x150 um^2
        psize = 2;
        if (xbin > xBinMax_[psize] || ybin > yBinMax_[psize]) {
          edm::LogError("PPS RPixChargeShare") << " Array index out of bounds 2";
          continue;
        }
      }
      if (pixel_width_x < 0.11 && pixel_width_y > 0.151) {  // pixel 100x300 um^2
        psize = 1;
        if (xbin > xBinMax_[psize] || ybin > yBinMax_[psize]) {
          edm::LogError("PPS RPixChargeShare") << " Array index out of bounds 1";
          continue;
        }
      }
      if (pixel_width_x > 0.11 && pixel_width_y > 0.151) {  // pixel 200x300 um^2
        psize = 3;
        if (xbin > xBinMax_[psize] || ybin > yBinMax_[psize]) {
          edm::LogError("PPS RPixChargeShare") << " Array index out of bounds 3";
          continue;
        }
      }
      if (xbin < 0 || ybin < 0) {
        edm::LogError("PPS RPixChargeShare") << " Negative array index xbin or ybin";
        continue;
      }
      double hit2neighbour[8];
      double collect_prob = chargeMap2E_[psize][xbin][ybin];
      thePixelChargeMap[pixel_no] += charge_in_pixel * collect_prob;
      unsigned short neighbour_no[8];
      unsigned short m = 0;
      double closer_neighbour = 0;
      unsigned short closer_no = 0;
      //      Considering the 8 neighbours to share charge
      for (int k = pixel_row - 1; k <= pixel_row + 1; k++) {
        for (int l = pixel_col - 1; l <= pixel_col + 1; l++) {
          if ((k < 0) || k > pxlRowSize_ - 1 || l < 0 || l > pxlColSize_ - 1)
            continue;
          if ((k == pixel_row) && (l == pixel_col))
            continue;
          double neighbour_pixel_lower_x = 0;
          double neighbour_pixel_lower_y = 0;
          double neighbour_pixel_upper_x = 0;
          double neighbour_pixel_upper_y = 0;
          double neighbour_pixel_center_x = 0;
          double neighbour_pixel_center_y = 0;
          //      Check the hit approach to the neighbours
          ppt.pixelRange(
              k, l, neighbour_pixel_lower_x, neighbour_pixel_upper_x, neighbour_pixel_lower_y, neighbour_pixel_upper_y);
          neighbour_pixel_center_x = neighbour_pixel_lower_x + (neighbour_pixel_upper_x - neighbour_pixel_lower_x) / 2.;
          neighbour_pixel_center_y = neighbour_pixel_lower_y + (neighbour_pixel_upper_y - neighbour_pixel_lower_y) / 2.;
          hit2neighbour[m] = sqrt(pow((*i).Position().x() - neighbour_pixel_center_x, 2.) +
                                  pow((*i).Position().y() - neighbour_pixel_center_y, -2.));
          neighbour_no[m] = l * pxlRowSize_ + k;
          if (hit2neighbour[m] > closer_neighbour) {
            closer_neighbour = hit2neighbour[m];
            closer_no = neighbour_no[m];
          }
          m++;
        }
      }
      double chargetransfereff = (1 - collect_prob) * signalCoupling_[0];
      thePixelChargeMap[closer_no] += charge_in_pixel * chargetransfereff;
    }
  }

  return thePixelChargeMap;
}
