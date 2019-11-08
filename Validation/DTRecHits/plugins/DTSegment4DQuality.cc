/*
 *  See header file for a description of this class.
 *
 *  \author S. Bolognesi and G. Cerminara - INFN Torino
 */

#include <iostream>
#include <map>

#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "Geometry/DTGeometry/interface/DTChamber.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "Validation/DTRecHits/interface/DTHitQualityUtils.h"

#include "DTSegment4DQuality.h"
#include "Histograms.h"

using namespace std;
using namespace edm;

namespace dtsegment4d {
  struct Histograms {
    std::unique_ptr<HRes4DHit> h4DHit;
    std::unique_ptr<HRes4DHit> h4DHit_W0;
    std::unique_ptr<HRes4DHit> h4DHit_W1;
    std::unique_ptr<HRes4DHit> h4DHit_W2;
    std::unique_ptr<HRes4DHit> h4DHitWS[3][4];

    std::unique_ptr<HEff4DHit> hEff_All;
    std::unique_ptr<HEff4DHit> hEff_W0;
    std::unique_ptr<HEff4DHit> hEff_W1;
    std::unique_ptr<HEff4DHit> hEff_W2;
    std::unique_ptr<HEff4DHit> hEffWS[3][4];
  };
}  // namespace dtsegment4d
using namespace dtsegment4d;

// In phi SLs, The dependency on X and angle is specular in positive
// and negative wheels. Since positive and negative wheels are filled
// together into the same plots, it is useful to mirror negative wheels
// so that the actual dependency can be observerd instead of an artificially
// simmetrized one.
// Set mirrorMinusWheels to avoid this.
namespace {
  constexpr bool mirrorMinusWheels = true;
}

// Constructor
DTSegment4DQuality::DTSegment4DQuality(const ParameterSet &pset) {
  // Get the debug parameter for verbose output
  debug_ = pset.getUntrackedParameter<bool>("debug");
  DTHitQualityUtils::debug = debug_;

  // the name of the simhit collection
  simHitLabel_ = pset.getUntrackedParameter<InputTag>("simHitLabel");
  simHitToken_ = consumes<PSimHitContainer>(pset.getUntrackedParameter<InputTag>("simHitLabel"));
  // the name of the 2D rec hit collection
  segment4DLabel_ = pset.getUntrackedParameter<InputTag>("segment4DLabel");
  segment4DToken_ = consumes<DTRecSegment4DCollection>(pset.getUntrackedParameter<InputTag>("segment4DLabel"));

  // sigma resolution on position
  sigmaResX_ = pset.getParameter<double>("sigmaResX");
  sigmaResY_ = pset.getParameter<double>("sigmaResY");
  // sigma resolution on angle
  sigmaResAlpha_ = pset.getParameter<double>("sigmaResAlpha");
  sigmaResBeta_ = pset.getParameter<double>("sigmaResBeta");
  doall_ = pset.getUntrackedParameter<bool>("doall", false);
  local_ = pset.getUntrackedParameter<bool>("local", false);
}

void DTSegment4DQuality::bookHistograms(DQMStore::IBooker &booker,
                                        edm::Run const &run,
                                        edm::EventSetup const &setup,
                                        Histograms &histograms) const {
  histograms.h4DHit = std::make_unique<HRes4DHit>("All", booker, doall_, local_);
  histograms.h4DHit_W0 = std::make_unique<HRes4DHit>("W0", booker, doall_, local_);
  histograms.h4DHit_W1 = std::make_unique<HRes4DHit>("W1", booker, doall_, local_);
  histograms.h4DHit_W2 = std::make_unique<HRes4DHit>("W2", booker, doall_, local_);

  if (doall_) {
    histograms.hEff_All = std::make_unique<HEff4DHit>("All", booker);
    histograms.hEff_W0 = std::make_unique<HEff4DHit>("W0", booker);
    histograms.hEff_W1 = std::make_unique<HEff4DHit>("W1", booker);
    histograms.hEff_W2 = std::make_unique<HEff4DHit>("W2", booker);
  }

  if (local_) {
    // Plots with finer granularity, not to be included in DQM
    TString name = "W";
    for (long w = 0; w <= 2; ++w) {
      for (long s = 1; s <= 4; ++s) {
        // FIXME station 4 is not filled
        TString nameWS = (name + w + "_St" + s);
        histograms.h4DHitWS[w][s - 1] = std::make_unique<HRes4DHit>(nameWS.Data(), booker, doall_, local_);
        histograms.hEffWS[w][s - 1] = std::make_unique<HEff4DHit>(nameWS.Data(), booker);
      }
    }
  }
};

// The real analysis
void DTSegment4DQuality::dqmAnalyze(edm::Event const &event,
                                    edm::EventSetup const &setup,
                                    Histograms const &histograms) const {
  const float epsilon = 5e-5;  // numerical accuracy on angles [rad}

  // Get the DT Geometry
  ESHandle<DTGeometry> dtGeom;
  setup.get<MuonGeometryRecord>().get(dtGeom);

  // Get the SimHit collection from the event
  edm::Handle<PSimHitContainer> simHits;
  event.getByToken(simHitToken_, simHits);  // FIXME: second string to be removed

  // Map simHits by chamber
  map<DTChamberId, PSimHitContainer> simHitsPerCh;
  for (const auto &simHit : *simHits) {
    // Consider only muon simhits; the others are not considered elsewhere in
    // this class!
    if (abs(simHit.particleType()) == 13) {
      // Create the id of the chamber (the simHits in the DT known their wireId)
      DTChamberId chamberId = (((DTWireId(simHit.detUnitId())).layerId()).superlayerId()).chamberId();
      // Fill the map
      simHitsPerCh[chamberId].push_back(simHit);
    }
  }

  // Get the 4D rechits from the event
  Handle<DTRecSegment4DCollection> segment4Ds;
  event.getByToken(segment4DToken_, segment4Ds);

  if (!segment4Ds.isValid()) {
    if (debug_) {
      cout << "[DTSegment4DQuality]**Warning: no 4D Segments with label: " << segment4DLabel_
           << " in this event, skipping!" << endl;
    }
    return;
  }

  // Loop over all chambers containing a (muon) simhit
  for (auto &simHitsInChamber : simHitsPerCh) {
    DTChamberId chamberId = simHitsInChamber.first;
    int station = chamberId.station();
    if (station == 4 && !(local_)) {
      continue;  // use DTSegment2DSLPhiQuality to analyze MB4 performaces in DQM
    }
    int wheel = chamberId.wheel();

    //------------------------- simHits ---------------------------//
    // Get simHits of this chamber
    const PSimHitContainer &simHits = simHitsInChamber.second;

    // Map simhits per wire
    auto const &simHitsPerWire = DTHitQualityUtils::mapSimHitsPerWire(simHits);
    auto const &muSimHitPerWire = DTHitQualityUtils::mapMuSimHitsPerWire(simHitsPerWire);
    int nMuSimHit = muSimHitPerWire.size();
    if (nMuSimHit < 2) {  // Skip chamber with less than 2 cells with mu hits
      continue;
    }
    if (debug_) {
      cout << "=== Chamber " << chamberId << " has " << nMuSimHit << " SimHits" << endl;
    }

    // Find outer and inner mu SimHit to build a segment
    pair<const PSimHit *, const PSimHit *> inAndOutSimHit = DTHitQualityUtils::findMuSimSegment(muSimHitPerWire);

    // Consider only sim segments crossing at least 2 SLs
    if ((DTWireId(inAndOutSimHit.first->detUnitId())).superlayer() ==
        (DTWireId(inAndOutSimHit.second->detUnitId())).superLayer()) {
      continue;
    }

    // Find direction and position of the sim Segment in Chamber RF
    pair<LocalVector, LocalPoint> dirAndPosSimSegm =
        DTHitQualityUtils::findMuSimSegmentDirAndPos(inAndOutSimHit, chamberId, &(*dtGeom));

    LocalVector simSegmLocalDir = dirAndPosSimSegm.first;
    LocalPoint simSegmLocalPos = dirAndPosSimSegm.second;
    const DTChamber *chamber = dtGeom->chamber(chamberId);
    GlobalPoint simSegmGlobalPos = chamber->toGlobal(simSegmLocalPos);
    GlobalVector simSegmGlobalDir = chamber->toGlobal(simSegmLocalDir);

    // phi and theta angle of simulated segment in Chamber RF
    float alphaSimSeg = DTHitQualityUtils::findSegmentAlphaAndBeta(simSegmLocalDir).first;
    float betaSimSeg = DTHitQualityUtils::findSegmentAlphaAndBeta(simSegmLocalDir).second;
    // x, y position of simulated segment in Chamber RF
    float xSimSeg = simSegmLocalPos.x();
    float ySimSeg = simSegmLocalPos.y();
    // Position (in eta, phi coordinates) in lobal RF
    float etaSimSeg = simSegmGlobalPos.eta();
    float phiSimSeg = simSegmGlobalPos.phi();

    double count_seg = 0;

    if (debug_) {
      cout << "  Simulated segment:  local direction " << simSegmLocalDir << endl
           << "                      local position  " << simSegmLocalPos << endl
           << "                      alpha           " << alphaSimSeg << endl
           << "                      beta            " << betaSimSeg << endl;
    }

    //---------------------------- recHits --------------------------//
    // Get the range of rechit for the corresponding chamberId
    bool recHitFound = false;
    DTRecSegment4DCollection::range range = segment4Ds->get(chamberId);
    int nsegm = distance(range.first, range.second);
    if (debug_) {
      cout << "   Chamber: " << chamberId << " has " << nsegm << " 4D segments" << endl;
    }

    if (nsegm != 0) {
      // Find the best RecHit: look for the 4D RecHit with the phi angle closest
      // to that of segment made of SimHits.
      // RecHits must have delta alpha and delta position within 5 sigma of
      // the residual distribution (we are looking for residuals of segments
      // usefull to the track fit) for efficency purpose
      const DTRecSegment4D *bestRecHit = nullptr;
      double deltaAlpha = 99999;
      double deltaBeta = 99999;

      // Loop over the recHits of this chamberId
      for (DTRecSegment4DCollection::const_iterator segment4D = range.first; segment4D != range.second; ++segment4D) {
        // Consider only segments with both projections
        if (station != 4 && (*segment4D).dimension() != 4) {
          continue;
        }
        // Segment Local Direction and position (in Chamber RF)
        LocalVector recSegDirection = (*segment4D).localDirection();
        LocalPoint recSegPosition = (*segment4D).localPosition();

        pair<double, double> ab = DTHitQualityUtils::findSegmentAlphaAndBeta(recSegDirection);
        float recSegAlpha = ab.first;
        float recSegBeta = ab.second;

        if (debug_) {
          cout << &(*segment4D) << "  RecSegment direction: " << recSegDirection << endl
               << "             position : " << (*segment4D).localPosition() << endl
               << "             alpha    : " << recSegAlpha << endl
               << "             beta     : " << recSegBeta << endl
               << "             nhits    : " << (*segment4D).phiSegment()->recHits().size() << " "
               << (((*segment4D).zSegment() != nullptr) ? (*segment4D).zSegment()->recHits().size() : 0) << endl;
        }

        float dAlphaRecSim = fabs(recSegAlpha - alphaSimSeg);
        float dBetaRecSim = fabs(recSegBeta - betaSimSeg);

        if ((fabs(recSegPosition.x() - simSegmLocalPos.x()) <
             4)  // require rec and sim segments to be ~in the same cell in x
            && ((fabs(recSegPosition.y() - simSegmLocalPos.y()) < 4) ||
                (*segment4D).dimension() < 4)) {  // ~in the same cell in y, if segment has the theta view
          ++count_seg;

          if (fabs(dAlphaRecSim - deltaAlpha) < epsilon) {  // Numerically equivalent alphas, choose based on beta
            if (dBetaRecSim < deltaBeta) {
              deltaAlpha = dAlphaRecSim;
              deltaBeta = dBetaRecSim;
              bestRecHit = &(*segment4D);
            }

          } else if (dAlphaRecSim < deltaAlpha) {
            deltaAlpha = dAlphaRecSim;
            deltaBeta = dBetaRecSim;
            bestRecHit = &(*segment4D);
          }
        }

      }  // End of Loop over all 4D RecHits

      if (bestRecHit) {
        if (debug_) {
          cout << endl << "Chosen: " << bestRecHit << endl;
        }
        // Best rechit direction and position in Chamber RF
        LocalPoint bestRecHitLocalPos = bestRecHit->localPosition();
        LocalVector bestRecHitLocalDir = bestRecHit->localDirection();
        // Errors on x and y
        LocalError bestRecHitLocalPosErr = bestRecHit->localPositionError();
        LocalError bestRecHitLocalDirErr = bestRecHit->localDirectionError();

        pair<double, double> ab = DTHitQualityUtils::findSegmentAlphaAndBeta(bestRecHitLocalDir);
        float alphaBestRHit = ab.first;
        float betaBestRHit = ab.second;
        // Errors on alpha and beta

        // Get position and direction using the rx projection (so in SL
        // reference frame). Note that x (and y) are swapped wrt to Chamber
        // frame
        // if (bestRecHit->hasZed()) //
        const DTSLRecSegment2D *zedRecSeg = bestRecHit->zSegment();
        LocalPoint bestRecHitLocalPosRZ;
        LocalVector bestRecHitLocalDirRZ;
        LocalError bestRecHitLocalPosErrRZ;
        LocalError bestRecHitLocalDirErrRZ;
        LocalPoint simSegLocalPosRZ;  // FIXME: this is not set for segments with
                                      // only the phi view.
        float alphaBestRHitRZ = 0;    // angle measured in the RZ SL, in its own frame
        float alphaSimSegRZ = betaSimSeg;
        if (zedRecSeg) {
          bestRecHitLocalPosRZ = zedRecSeg->localPosition();
          bestRecHitLocalDirRZ = zedRecSeg->localDirection();
          // Errors on x and y
          bestRecHitLocalPosErrRZ = zedRecSeg->localPositionError();
          bestRecHitLocalDirErrRZ = zedRecSeg->localDirectionError();

          // angle measured in the RZ SL, in its own frame
          alphaBestRHitRZ = DTHitQualityUtils::findSegmentAlphaAndBeta(bestRecHitLocalDirRZ).first;

          // Get SimSeg position and Direction in rZ SL frame
          const DTSuperLayer *sl = dtGeom->superLayer(zedRecSeg->superLayerId());
          LocalPoint simSegLocalPosRZTmp = sl->toLocal(simSegmGlobalPos);
          LocalVector simSegLocalDirRZ = sl->toLocal(simSegmGlobalDir);
          simSegLocalPosRZ =
              simSegLocalPosRZTmp + simSegLocalDirRZ * (-simSegLocalPosRZTmp.z() / (cos(simSegLocalDirRZ.theta())));
          alphaSimSegRZ = DTHitQualityUtils::findSegmentAlphaAndBeta(simSegLocalDirRZ).first;

          if (debug_) {
            cout << "RZ SL: recPos " << bestRecHitLocalPosRZ << "recDir " << bestRecHitLocalDirRZ << "recAlpha "
                 << alphaBestRHitRZ << endl
                 << "RZ SL: simPos " << simSegLocalPosRZ << "simDir " << simSegLocalDirRZ << "simAlpha "
                 << alphaSimSegRZ << endl;
          }
        }

        // get nhits and t0
        const DTChamberRecSegment2D *phiSeg = bestRecHit->phiSegment();

        float t0phi = -999;
        float t0theta = -999;
        int nHitPhi = 0;
        int nHitTheta = 0;

        if (phiSeg) {
          t0phi = phiSeg->t0();
          nHitPhi = phiSeg->recHits().size();
        }

        if (zedRecSeg) {
          t0theta = zedRecSeg->t0();
          nHitTheta = zedRecSeg->recHits().size();
        }

        recHitFound = true;

        // Mirror alpha in phi SLs so that + and - wheels can be plotted
        // together
        if (mirrorMinusWheels && wheel < 0) {
          alphaSimSeg *= -1.;
          alphaBestRHit *= -1.;
          // Note: local X (xSimSeg, bestRecHitLocalPos.x() would have to be
          // mirrored as well; but at the moment there is no plot of dependency
          // vs X, except for efficiency.
        }

        // Fill Residual histos
        HRes4DHit *histo = nullptr;

        if (wheel == 0) {
          histo = histograms.h4DHit_W0.get();
        } else if (abs(wheel) == 1) {
          histo = histograms.h4DHit_W1.get();
        } else if (abs(wheel) == 2) {
          histo = histograms.h4DHit_W2.get();
        }

        float sigmaAlphaBestRhit = sqrt(DTHitQualityUtils::sigmaAngle(alphaBestRHit, bestRecHitLocalDirErr.xx()));
        float sigmaBetaBestRhit =
            sqrt(DTHitQualityUtils::sigmaAngle(betaBestRHit,
                                               bestRecHitLocalDirErr.yy()));  // FIXME this misses the contribution
                                                                              // from uncertainty in extrapolation!
        float sigmaAlphaBestRhitRZ = sqrt(DTHitQualityUtils::sigmaAngle(alphaBestRHitRZ, bestRecHitLocalDirErrRZ.xx()));

        histo->fill(alphaSimSeg,
                    alphaBestRHit,
                    betaSimSeg,
                    betaBestRHit,
                    xSimSeg,
                    bestRecHitLocalPos.x(),
                    ySimSeg,
                    bestRecHitLocalPos.y(),
                    etaSimSeg,
                    phiSimSeg,
                    bestRecHitLocalPosRZ.x(),
                    simSegLocalPosRZ.x(),
                    alphaBestRHitRZ,
                    alphaSimSegRZ,
                    sigmaAlphaBestRhit,
                    sigmaBetaBestRhit,
                    sqrt(bestRecHitLocalPosErr.xx()),
                    sqrt(bestRecHitLocalPosErr.yy()),
                    sigmaAlphaBestRhitRZ,
                    sqrt(bestRecHitLocalPosErrRZ.xx()),
                    nHitPhi,
                    nHitTheta,
                    t0phi,
                    t0theta);

        histograms.h4DHit->fill(alphaSimSeg,
                                alphaBestRHit,
                                betaSimSeg,
                                betaBestRHit,
                                xSimSeg,
                                bestRecHitLocalPos.x(),
                                ySimSeg,
                                bestRecHitLocalPos.y(),
                                etaSimSeg,
                                phiSimSeg,
                                bestRecHitLocalPosRZ.x(),
                                simSegLocalPosRZ.x(),
                                alphaBestRHitRZ,
                                alphaSimSegRZ,
                                sigmaAlphaBestRhit,
                                sigmaBetaBestRhit,
                                sqrt(bestRecHitLocalPosErr.xx()),
                                sqrt(bestRecHitLocalPosErr.yy()),
                                sigmaAlphaBestRhitRZ,
                                sqrt(bestRecHitLocalPosErrRZ.xx()),
                                nHitPhi,
                                nHitTheta,
                                t0phi,
                                t0theta);

        if (local_) {
          histograms.h4DHitWS[abs(wheel)][station - 1]->fill(alphaSimSeg,
                                                             alphaBestRHit,
                                                             betaSimSeg,
                                                             betaBestRHit,
                                                             xSimSeg,
                                                             bestRecHitLocalPos.x(),
                                                             ySimSeg,
                                                             bestRecHitLocalPos.y(),
                                                             etaSimSeg,
                                                             phiSimSeg,
                                                             bestRecHitLocalPosRZ.x(),
                                                             simSegLocalPosRZ.x(),
                                                             alphaBestRHitRZ,
                                                             alphaSimSegRZ,
                                                             sigmaAlphaBestRhit,
                                                             sigmaBetaBestRhit,
                                                             sqrt(bestRecHitLocalPosErr.xx()),
                                                             sqrt(bestRecHitLocalPosErr.yy()),
                                                             sigmaAlphaBestRhitRZ,
                                                             sqrt(bestRecHitLocalPosErrRZ.xx()),
                                                             nHitPhi,
                                                             nHitTheta,
                                                             t0phi,
                                                             t0theta);
        }

      }  // end of if (bestRecHit)

    }  // end of if (nsegm!= 0)

    // Fill Efficiency plot
    if (doall_) {
      HEff4DHit *heff = nullptr;

      if (wheel == 0) {
        heff = histograms.hEff_W0.get();
      } else if (abs(wheel) == 1) {
        heff = histograms.hEff_W1.get();
      } else if (abs(wheel) == 2) {
        heff = histograms.hEff_W2.get();
      }
      heff->fill(etaSimSeg, phiSimSeg, xSimSeg, ySimSeg, alphaSimSeg, betaSimSeg, recHitFound, count_seg);
      histograms.hEff_All->fill(
          etaSimSeg, phiSimSeg, xSimSeg, ySimSeg, alphaSimSeg, betaSimSeg, recHitFound, count_seg);
      if (local_) {
        histograms.hEffWS[abs(wheel)][station - 1]->fill(
            etaSimSeg, phiSimSeg, xSimSeg, ySimSeg, alphaSimSeg, betaSimSeg, recHitFound, count_seg);
      }
    }
  }  // End of loop over chambers
}

// declare this as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DTSegment4DQuality);
