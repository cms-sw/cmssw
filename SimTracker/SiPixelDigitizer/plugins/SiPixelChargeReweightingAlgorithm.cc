//class SiPixelChargeReweightingAlgorithm SimTracker/SiPixelDigitizer/src/SiPixelChargeReweightingAlgorithm.cc

// Original Author Caroline Collard
// September 2020 : Extraction of the code for cluster charge reweighting from SiPixelDigitizerAlgorithm to a new class
//
#include <iostream>
#include <iomanip>

#include "SimGeneral/NoiseGenerators/interface/GaussianTailNoiseGenerator.h"

#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimTracker/Common/interface/SiG4UniversalFluctuation.h"
#include "SimTracker/SiPixelDigitizer/plugins/SiPixelChargeReweightingAlgorithm.h"

//#include "PixelIndices.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// Accessing dead pixel modules from the DB:
#include "DataFormats/DetId/interface/DetId.h"

#include "CondFormats/SiPixelObjects/interface/GlobalPixel.h"

#include "CondFormats/SiPixelObjects/interface/PixelIndices.h"
#include "CondFormats/SiPixelObjects/interface/PixelROC.h"
#include "CondFormats/SiPixelObjects/interface/LocalPixel.h"
#include "CondFormats/SiPixelObjects/interface/SiPixel2DTemplateDBObject.h"

#include "CondFormats/SiPixelObjects/interface/PixelROC.h"

using namespace edm;
using namespace sipixelobjects;

void SiPixelChargeReweightingAlgorithm::init(const edm::EventSetup& es) {
  // Read template files for charge reweighting
  if (UseReweighting) {
    dbobject_den = &es.getData(SiPixel2DTemp_den_token_);
    dbobject_num = &es.getData(SiPixel2DTemp_num_token_);

    int numOfTemplates = dbobject_den->numOfTempl() + dbobject_num->numOfTempl();
    templateStores_.reserve(numOfTemplates);
    SiPixelTemplate2D::pushfile(*dbobject_den, templateStores_);
    SiPixelTemplate2D::pushfile(*dbobject_num, templateStores_);

    track.resize(6);
  }
}

//=========================================================================

SiPixelChargeReweightingAlgorithm::SiPixelChargeReweightingAlgorithm(const edm::ParameterSet& conf,
                                                                     edm::ConsumesCollector iC)
    :

      templ2D(templateStores_),
      xdouble(TXSIZE),
      ydouble(TYSIZE),
      IDnum(conf.exists("TemplateIDnumerator") ? conf.getParameter<int>("TemplateIDnumerator") : 0),
      IDden(conf.exists("TemplateIDdenominator") ? conf.getParameter<int>("TemplateIDdenominator") : 0),

      UseReweighting(conf.getParameter<bool>("UseReweighting")),
      PrintClusters(conf.getParameter<bool>("PrintClusters")),
      PrintTemplates(conf.getParameter<bool>("PrintTemplates")) {
  if (UseReweighting) {
    SiPixel2DTemp_den_token_ = iC.esConsumes(edm::ESInputTag("", "denominator"));
    SiPixel2DTemp_num_token_ = iC.esConsumes(edm::ESInputTag("", "numerator"));
  }
  edm::LogVerbatim("PixelDigitizer ") << "SiPixelChargeReweightingAlgorithm constructed"
                                      << " with UseReweighting = " << UseReweighting;
}

//=========================================================================
SiPixelChargeReweightingAlgorithm::~SiPixelChargeReweightingAlgorithm() {
  LogDebug("PixelDigitizer") << "SiPixelChargeReweightingAlgorithm deleted";
}

//============================================================================

bool SiPixelChargeReweightingAlgorithm::hitSignalReweight(const PSimHit& hit,
                                                          std::map<int, float, std::less<int> >& hit_signal,
                                                          const size_t hitIndex,
                                                          const unsigned int tofBin,
                                                          const PixelTopology* topol,
                                                          uint32_t detID,
                                                          signal_map_type& theSignal,
                                                          unsigned short int processType,
                                                          const bool& boolmakeDigiSimLinks) {
  int irow_min = topol->nrows();
  int irow_max = 0;
  int icol_min = topol->ncolumns();
  int icol_max = 0;

  float chargeBefore = 0;
  float chargeAfter = 0;
  signal_map_type hitSignal;
  LocalVector direction = hit.exitPoint() - hit.entryPoint();

  for (std::map<int, float, std::less<int> >::const_iterator im = hit_signal.begin(); im != hit_signal.end(); ++im) {
    int chan = (*im).first;
    std::pair<int, int> pixelWithCharge = PixelDigi::channelToPixel(chan);

    hitSignal[chan] +=
        (boolmakeDigiSimLinks ? SiPixelDigitizerAlgorithm::Amplitude((*im).second, &hit, hitIndex, tofBin, (*im).second)
                              : SiPixelDigitizerAlgorithm::Amplitude((*im).second, (*im).second));
    chargeBefore += (*im).second;

    if (pixelWithCharge.first < irow_min)
      irow_min = pixelWithCharge.first;
    if (pixelWithCharge.first > irow_max)
      irow_max = pixelWithCharge.first;
    if (pixelWithCharge.second < icol_min)
      icol_min = pixelWithCharge.second;
    if (pixelWithCharge.second > icol_max)
      icol_max = pixelWithCharge.second;
  }

  LocalPoint hitEntryPoint = hit.entryPoint();

  float trajectoryScaleToPosition = hitEntryPoint.z() / direction.z();

  if ((hitEntryPoint.z() > 0 && direction.z() < 0) || (hitEntryPoint.z() < 0 && direction.z() > 0)) {
    trajectoryScaleToPosition *= -1;
  }

  LocalPoint hitPosition = hitEntryPoint + trajectoryScaleToPosition * direction;

  MeasurementPoint hitPositionPixel = topol->measurementPosition(hit.localPosition());
  std::pair<int, int> hitPixel =
      std::pair<int, int>(int(floor(hitPositionPixel.x())), int(floor(hitPositionPixel.y())));

  MeasurementPoint originPixel = MeasurementPoint(hitPixel.first - THX + 0.5, hitPixel.second - THY + 0.5);
  LocalPoint origin = topol->localPosition(originPixel);

  MeasurementPoint hitEntryPointPixel = topol->measurementPosition(hit.entryPoint());
  MeasurementPoint hitExitPointPixel = topol->measurementPosition(hit.exitPoint());
  std::pair<int, int> entryPixel =
      std::pair<int, int>(int(floor(hitEntryPointPixel.x())), int(floor(hitEntryPointPixel.y())));
  std::pair<int, int> exitPixel =
      std::pair<int, int>(int(floor(hitExitPointPixel.x())), int(floor(hitExitPointPixel.y())));

  int hitcol_min, hitcol_max, hitrow_min, hitrow_max;
  if (entryPixel.first > exitPixel.first) {
    hitrow_min = exitPixel.first;
    hitrow_max = entryPixel.first;
  } else {
    hitrow_min = entryPixel.first;
    hitrow_max = exitPixel.first;
  }

  if (entryPixel.second > exitPixel.second) {
    hitcol_min = exitPixel.second;
    hitcol_max = entryPixel.second;
  } else {
    hitcol_min = entryPixel.second;
    hitcol_max = exitPixel.second;
  }

#ifdef TP_DEBUG
  LocalPoint CMSSWhitPosition = hit.localPosition();

  LogDebug("Pixel Digitizer") << "\n"
                              << "Particle ID is: " << hit.particleType() << "\n"
                              << "Process type: " << hit.processType() << "\n"
                              << "HitPosition:"
                              << "\n"
                              << "Hit entry x/y/z: " << hit.entryPoint().x() << "  " << hit.entryPoint().y() << "  "
                              << hit.entryPoint().z() << "  "
                              << "Hit exit x/y/z: " << hit.exitPoint().x() << "  " << hit.exitPoint().y() << "  "
                              << hit.exitPoint().z() << "  "

                              << "Pixel Pos - X: " << hitPositionPixel.x() << " Y: " << hitPositionPixel.y() << "\n"
                              << "Cart.Cor. - X: " << CMSSWhitPosition.x() << " Y: " << CMSSWhitPosition.y() << "\n"
                              << "Z=0 Pos - X: " << hitPosition.x() << " Y: " << hitPosition.y() << "\n"

                              << "Origin of the template:"
                              << "\n"
                              << "Pixel Pos - X: " << originPixel.x() << " Y: " << originPixel.y() << "\n"
                              << "Cart.Cor. - X: " << origin.x() << " Y: " << origin.y() << "\n"
                              << "\n"
                              << "Entry/Exit:"
                              << "\n"
                              << "Entry - X: " << hit.entryPoint().x() << " Y: " << hit.entryPoint().y()
                              << " Z: " << hit.entryPoint().z() << "\n"
                              << "Exit - X: " << hit.exitPoint().x() << " Y: " << hit.exitPoint().y()
                              << " Z: " << hit.exitPoint().z() << "\n"

                              << "Entry - X Pixel: " << hitEntryPointPixel.x() << " Y Pixel: " << hitEntryPointPixel.y()
                              << "\n"
                              << "Exit - X Pixel: " << hitExitPointPixel.x() << " Y Pixel: " << hitExitPointPixel.y()
                              << "\n"

                              << "row min: " << irow_min << " col min: " << icol_min << "\n";
#endif

  if (!(irow_min <= hitrow_max && irow_max >= hitrow_min && icol_min <= hitcol_max && icol_max >= hitcol_min)) {
    // The clusters do not have an overlap, hence the hit is NOT reweighted
    return false;
  }

  float cmToMicrons = 10000.f;

  track[0] = (hitPosition.x() - origin.x()) * cmToMicrons;
  track[1] = (hitPosition.y() - origin.y()) * cmToMicrons;
  track[2] = 0.0f;  //Middle of sensor is origin for Z-axis
  track[3] = direction.x();
  track[4] = direction.y();
  track[5] = direction.z();

  array_2d pixrewgt(boost::extents[TXSIZE][TYSIZE]);

  for (int row = 0; row < TXSIZE; ++row) {
    for (int col = 0; col < TYSIZE; ++col) {
      pixrewgt[row][col] = 0;
    }
  }

  for (int row = 0; row < TXSIZE; ++row) {
    xdouble[row] = topol->isItBigPixelInX(hitPixel.first + row - THX);
  }

  for (int col = 0; col < TYSIZE; ++col) {
    ydouble[col] = topol->isItBigPixelInY(hitPixel.second + col - THY);
  }

  for (int row = 0; row < TXSIZE; ++row) {
    for (int col = 0; col < TYSIZE; ++col) {
      //Fill charges into 21x13 Pixel Array with hitPixel in centre
      pixrewgt[row][col] =
          hitSignal[PixelDigi::pixelToChannel(hitPixel.first + row - THX, hitPixel.second + col - THY)];
    }
  }

  if (PrintClusters) {
    std::cout << "Cluster before reweighting: " << std::endl;
    printCluster(pixrewgt);
  }

  int ierr;
  // for unirradiated: 2nd argument is IDden
  // for irradiated: 2nd argument is IDnum
  if (UseReweighting == true) {
    int ID1 = dbobject_num->getTemplateID(detID);
    int ID0 = dbobject_den->getTemplateID(detID);

    if (ID0 == ID1) {
      return false;
    }
    ierr = PixelTempRewgt2D(ID0, ID1, pixrewgt);
  } else {
    ierr = PixelTempRewgt2D(IDden, IDden, pixrewgt);
  }
  if (ierr != 0) {
#ifdef TP_DEBUG
    LogDebug("PixelDigitizer ") << "Cluster Charge Reweighting did not work properly.";
#endif
    return false;
  }

  if (PrintClusters) {
    std::cout << "Cluster after reweighting: " << std::endl;
    printCluster(pixrewgt);
  }

  for (int row = 0; row < TXSIZE; ++row) {
    for (int col = 0; col < TYSIZE; ++col) {
      float charge = 0;
      charge = pixrewgt[row][col];
      if ((hitPixel.first + row - THX) >= 0 && (hitPixel.first + row - THX) < topol->nrows() &&
          (hitPixel.second + col - THY) >= 0 && (hitPixel.second + col - THY) < topol->ncolumns() && charge > 0) {
        chargeAfter += charge;
        theSignal[PixelDigi::pixelToChannel(hitPixel.first + row - THX, hitPixel.second + col - THY)] +=
            (boolmakeDigiSimLinks ? SiPixelDigitizerAlgorithm::Amplitude(charge, &hit, hitIndex, tofBin, charge)
                                  : SiPixelDigitizerAlgorithm::Amplitude(charge, charge));
      }
    }
  }

  if (chargeBefore != 0. && chargeAfter == 0.) {
    return false;
  }

  if (PrintClusters) {
    std::cout << std::endl;
    std::cout << "Charges (before->after): " << chargeBefore << " -> " << chargeAfter << std::endl;
    std::cout << "Charge loss: " << (1 - chargeAfter / chargeBefore) * 100 << " %" << std::endl << std::endl;
  }

  return true;
}

// *******************************************************************************************************
//! Reweight CMSSW clusters to look like clusters corresponding to Pixelav Templates.
//! \param       id_in - (input) identifier of the template corresponding to the input events
//! \param    id_rewgt - (input) identifier of the template corresponding to the output events
//! \param     cluster - (input/output) boost multi_array container of 7x21 array of pixel signals,
//!                       origin of local coords (0,0) at center of pixel cluster[3][10].
//! returns 0 if everything is OK, 1 if angles are outside template coverage (cluster is probably still
//! usable, > 1 if something is wrong (no reweight done).
// *******************************************************************************************************
int SiPixelChargeReweightingAlgorithm::PixelTempRewgt2D(int id_in, int id_rewgt, array_2d& cluster) {
  // Local variables
  int i, j, k, l, kclose;
  int nclusx, nclusy, success;
  float xsize, ysize, q50i, q100i, q50r, q10r, q100r, xhit2D, yhit2D, qclust, dist2, dmin2;
  float xy_in[BXM2][BYM2], xy_rewgt[BXM2][BYM2], xy_clust[TXSIZE][TYSIZE];
  int denx_clust[TXSIZE][TYSIZE], deny_clust[TXSIZE][TYSIZE];
  int goodWeightsUsed, nearbyWeightsUsed, noWeightsUsed;
  float cotalpha, cotbeta;
  // success = 0 is returned if everthing is OK
  success = 0;

  // Copy the array to remember original charges
  array_2d clust(cluster);

  // Take the pixel dimensions from the 2D template
  templ2D.getid(id_in);
  xsize = templ2D.xsize();
  ysize = templ2D.ysize();

  // Calculate the track angles

  if (std::abs(track[5]) > 0.f) {
    cotalpha = track[3] / track[5];  //if track[5] (direction in z) is 0 the hit is not processed by re-weighting
    cotbeta = track[4] / track[5];
  } else {
    LogDebug("Pixel Digitizer") << "Reweighting angle is not good!" << std::endl;
    return 9;  //returned value here indicates that no reweighting was done in this case
  }

  // The 2-D templates are defined on a shifted coordinate system wrt the 1D templates
  if (ydouble[0]) {
    yhit2D = track[1] - cotbeta * track[2] + ysize;
  } else {
    yhit2D = track[1] - cotbeta * track[2] + 0.5f * ysize;
  }
  if (xdouble[0]) {
    xhit2D = track[0] - cotalpha * track[2] + xsize;
  } else {
    xhit2D = track[0] - cotalpha * track[2] + 0.5f * xsize;
  }

  // Zero the input and output templates
  for (i = 0; i < BYM2; ++i) {
    for (j = 0; j < BXM2; ++j) {
      xy_in[j][i] = 0.f;
      xy_rewgt[j][i] = 0.f;
    }
  }

  // Next, interpolate the CMSSW template needed to analyze this cluster

  if (!templ2D.xytemp(id_in, cotalpha, cotbeta, xhit2D, yhit2D, ydouble, xdouble, xy_in)) {
    success = 1;
  }
  if (success != 0) {
#ifdef TP_DEBUG
    LogDebug("Pixel Digitizer") << "No matching template found" << std::endl;
#endif
    return 2;
  }

  if (PrintTemplates) {
    std::cout << "Template unirrad: " << std::endl;
    printCluster(xy_in);
  }

  q50i = templ2D.s50();
  //q50i = 0;
  q100i = 2.f * q50i;

  // Check that the cluster container is a 13x21 matrix

  if (cluster.num_dimensions() != 2) {
    LogWarning("Pixel Digitizer") << "Cluster is not 2-dimensional. Return." << std::endl;
    return 3;
  }
  nclusx = (int)cluster.shape()[0];
  nclusy = (int)cluster.shape()[1];
  if (nclusx != TXSIZE || xdouble.size() != TXSIZE) {
    LogWarning("Pixel Digitizer") << "Sizes in x do not match: nclusx=" << nclusx << "  xdoubleSize=" << xdouble.size()
                                  << "  TXSIZE=" << TXSIZE << ". Return." << std::endl;
    return 4;
  }
  if (nclusy != TYSIZE || ydouble.size() != TYSIZE) {
    LogWarning("Pixel Digitizer") << "Sizes in y do not match. Return." << std::endl;
    return 5;
  }

  // Sum initial charge in the cluster

  qclust = 0.f;
  for (i = 0; i < TYSIZE; ++i) {
    for (j = 0; j < TXSIZE; ++j) {
      xy_clust[j][i] = 0.f;
      denx_clust[j][i] = 0;
      deny_clust[j][i] = 0;
      if (cluster[j][i] > q100i) {
        qclust += cluster[j][i];
      }
    }
  }

  // Next, interpolate the physical output template needed to reweight

  if (!templ2D.xytemp(id_rewgt, cotalpha, cotbeta, xhit2D, yhit2D, ydouble, xdouble, xy_rewgt)) {
    success = 1;
  }

  if (PrintTemplates) {
    std::cout << "Template irrad: " << std::endl;
    printCluster(xy_rewgt);
  }

  q50r = templ2D.s50();
  q100r = 2.f * q50r;
  q10r = 0.2f * q50r;

  // Find all non-zero denominator pixels in the input template and generate "inside" weights

  int ntpix = 0;
  int ncpix = 0;
  std::vector<int> ytclust;
  std::vector<int> xtclust;
  std::vector<int> ycclust;
  std::vector<int> xcclust;
  qclust = 0.f;
  for (i = 0; i < TYSIZE; ++i) {
    for (j = 0; j < TXSIZE; ++j) {
      if (xy_in[j + 1][i + 1] > q100i) {
        ++ntpix;
        ytclust.push_back(i);
        xtclust.push_back(j);
        xy_clust[j][i] = xy_rewgt[j + 1][i + 1] / xy_in[j + 1][i + 1];
        denx_clust[j][i] = j;
        deny_clust[j][i] = i;
      }
    }
  }

  // Find all non-zero numerator pixels not matched to denominator in the output template and generate "inside" weights

  for (i = 0; i < TYSIZE; ++i) {
    for (j = 0; j < TXSIZE; ++j) {
      if (xy_rewgt[j + 1][i + 1] > q10r && xy_clust[j][i] == 0.f && ntpix > 0) {
        // Search for nearest denominator pixel
        dmin2 = 10000.f;
        kclose = 0;
        for (k = 0; k < ntpix; ++k) {
          dist2 = (i - ytclust[k]) * (i - ytclust[k]) + 0.44444f * (j - xtclust[k]) * (j - xtclust[k]);
          if (dist2 < dmin2) {
            dmin2 = dist2;
            kclose = k;
          }
        }
        xy_clust[j][i] = xy_rewgt[j + 1][i + 1] / xy_in[xtclust[kclose] + 1][ytclust[kclose] + 1];
        denx_clust[j][i] = xtclust[kclose];
        deny_clust[j][i] = ytclust[kclose];
      }
    }
  }

  if (PrintTemplates) {
    std::cout << "Weights:" << std::endl;
    printCluster(xy_clust);
  }

  // Do the reweighting
  goodWeightsUsed = 0;
  nearbyWeightsUsed = 0;
  noWeightsUsed = 0;

  for (i = 0; i < TYSIZE; ++i) {
    for (j = 0; j < TXSIZE; ++j) {
      if (xy_clust[j][i] > 0.f) {
        cluster[j][i] = xy_clust[j][i] * clust[denx_clust[j][i]][deny_clust[j][i]];
        if (cluster[j][i] > q100r) {
          qclust += cluster[j][i];
        }
        if (cluster[j][i] > 0) {
          goodWeightsUsed++;
        }
      } else {
        if (clust[j][i] > 0.f) {
          ++ncpix;
          ycclust.push_back(i);
          xcclust.push_back(j);
        }
      }
    }
  }

  // Now reweight pixels outside of template footprint using closest weights

  if (ncpix > 0) {
    for (l = 0; l < ncpix; ++l) {
      i = ycclust[l];
      j = xcclust[l];
      dmin2 = 10000.f;
      kclose = 0;
      for (k = 0; k < ntpix; ++k) {
        dist2 = (i - ytclust[k]) * (i - ytclust[k]) + 0.44444f * (j - xtclust[k]) * (j - xtclust[k]);
        if (dist2 < dmin2) {
          dmin2 = dist2;
          kclose = k;
        }
      }
      if (dmin2 < 5.f) {
        nearbyWeightsUsed++;
        cluster[j][i] *= xy_clust[xtclust[kclose]][ytclust[kclose]];
        if (cluster[j][i] > q100r) {
          qclust += cluster[j][i];
        }
      } else {
        noWeightsUsed++;
        cluster[j][i] = 0.f;
      }
    }
  }

  return success;
}  // PixelTempRewgt2D

void SiPixelChargeReweightingAlgorithm::printCluster(array_2d& cluster) {
  for (int col = 0; col < TYSIZE; ++col) {
    for (int row = 0; row < TXSIZE; ++row) {
      std::cout << std::setw(10) << std::setprecision(0) << std::fixed;
      std::cout << cluster[row][col];
    }
    std::cout << std::endl;
  }
  std::cout.copyfmt(std::ios(nullptr));
}

void SiPixelChargeReweightingAlgorithm::printCluster(float arr[BXM2][BYM2]) {
  for (int col = 0; col < BYM2; ++col) {
    for (int row = 0; row < BXM2; ++row) {
      std::cout << std::setw(10) << std::setprecision(0) << std::fixed;
      std::cout << arr[row][col];
    }
    std::cout << std::endl;
  }
  std::cout.copyfmt(std::ios(nullptr));
}

void SiPixelChargeReweightingAlgorithm::printCluster(float arr[TXSIZE][TYSIZE]) {
  for (int col = 0; col < TYSIZE; ++col) {
    for (int row = 0; row < TXSIZE; ++row) {
      std::cout << std::setw(10) << std::fixed;
      std::cout << arr[row][col];
    }
    std::cout << std::endl;
  }
  std::cout.copyfmt(std::ios(nullptr));
}
