/**_________________________________________________________________
   class:   PVFitter.cc
   package: RecoVertex/BeamSpotProducer



   author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)
           Geng-Yuan Jeng, UC Riverside (Geng-Yuan.Jeng@cern.ch)


________________________________________________________________**/

#include "RecoVertex/BeamSpotProducer/interface/PVFitter.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "RecoVertex/BeamSpotProducer/interface/FcnBeamSpotFitPV.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include "Minuit2/FCNBase.h"
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnMigrad.h"
#include "Minuit2/MnPrint.h"
#include "TF1.h"
#include "TDirectory.h"
#include "TMinuitMinimizer.h"

#include <iostream>
#include <sstream>
using namespace std;

PVFitter::PVFitter(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iColl) : ftree_(nullptr) {
  initialize(iConfig, iColl);
}

PVFitter::PVFitter(const edm::ParameterSet& iConfig, edm::ConsumesCollector& iColl) : ftree_(nullptr) {
  initialize(iConfig, iColl);
}

void PVFitter::initialize(const edm::ParameterSet& iConfig, edm::ConsumesCollector& iColl) {
  //In order to make fitting ROOT histograms thread safe
  // one must call this undocumented function
  TMinuitMinimizer::UseStaticMinuit(false);
  debug_ = iConfig.getParameter<edm::ParameterSet>("PVFitter").getUntrackedParameter<bool>("Debug");
  vertexToken_ = iColl.consumes<reco::VertexCollection>(
      iConfig.getParameter<edm::ParameterSet>("PVFitter")
          .getUntrackedParameter<edm::InputTag>("VertexCollection", edm::InputTag("offlinePrimaryVertices")));
  do3DFit_ = iConfig.getParameter<edm::ParameterSet>("PVFitter").getUntrackedParameter<bool>("Apply3DFit");
  //writeTxt_          = iConfig.getParameter<edm::ParameterSet>("PVFitter").getUntrackedParameter<bool>("WriteAscii");
  //outputTxt_         = iConfig.getParameter<edm::ParameterSet>("PVFitter").getUntrackedParameter<std::string>("AsciiFileName");
  maxNrVertices_ =
      iConfig.getParameter<edm::ParameterSet>("PVFitter").getUntrackedParameter<unsigned int>("maxNrStoredVertices");
  minNrVertices_ =
      iConfig.getParameter<edm::ParameterSet>("PVFitter").getUntrackedParameter<unsigned int>("minNrVerticesForFit");
  minVtxNdf_ = iConfig.getParameter<edm::ParameterSet>("PVFitter").getUntrackedParameter<double>("minVertexNdf");
  maxVtxNormChi2_ =
      iConfig.getParameter<edm::ParameterSet>("PVFitter").getUntrackedParameter<double>("maxVertexNormChi2");
  minVtxTracks_ =
      iConfig.getParameter<edm::ParameterSet>("PVFitter").getUntrackedParameter<unsigned int>("minVertexNTracks");
  minVtxWgt_ = iConfig.getParameter<edm::ParameterSet>("PVFitter").getUntrackedParameter<double>("minVertexMeanWeight");
  maxVtxR_ = iConfig.getParameter<edm::ParameterSet>("PVFitter").getUntrackedParameter<double>("maxVertexR");
  maxVtxZ_ = iConfig.getParameter<edm::ParameterSet>("PVFitter").getUntrackedParameter<double>("maxVertexZ");
  errorScale_ = iConfig.getParameter<edm::ParameterSet>("PVFitter").getUntrackedParameter<double>("errorScale");
  sigmaCut_ = iConfig.getParameter<edm::ParameterSet>("PVFitter").getUntrackedParameter<double>("nSigmaCut");
  fFitPerBunchCrossing =
      iConfig.getParameter<edm::ParameterSet>("PVFitter").getUntrackedParameter<bool>("FitPerBunchCrossing");
  useOnlyFirstPV_ = iConfig.getParameter<edm::ParameterSet>("PVFitter").getUntrackedParameter<bool>("useOnlyFirstPV");
  minSumPt_ = iConfig.getParameter<edm::ParameterSet>("PVFitter").getUntrackedParameter<double>("minSumPt");

  // preset quality cut to "infinite"
  dynamicQualityCut_ = 1.e30;

  hPVx = std::unique_ptr<TH2F>(
      new TH2F("hPVx", "PVx vs PVz distribution", 200, -maxVtxR_, maxVtxR_, 200, -maxVtxZ_, maxVtxZ_));
  hPVy = std::unique_ptr<TH2F>(
      new TH2F("hPVy", "PVy vs PVz distribution", 200, -maxVtxR_, maxVtxR_, 200, -maxVtxZ_, maxVtxZ_));
  hPVx->SetDirectory(nullptr);
  hPVy->SetDirectory(nullptr);
}

PVFitter::~PVFitter() {}

void PVFitter::readEvent(const edm::Event& iEvent) {
  //------ Primary Vertices
  edm::Handle<reco::VertexCollection> PVCollection;
  bool hasPVs = false;

  if (iEvent.getByToken(vertexToken_, PVCollection)) {
    hasPVs = true;
  }
  //------

  if (hasPVs) {
    for (reco::VertexCollection::const_iterator pv = PVCollection->begin(); pv != PVCollection->end(); ++pv) {
      if (useOnlyFirstPV_) {
        if (pv != PVCollection->begin())
          break;
      }

      //--- vertex selection
      if (pv->isFake() || pv->tracksSize() == 0)
        continue;
      if (pv->ndof() < minVtxNdf_ || (pv->ndof() + 3.) / pv->tracksSize() < 2 * minVtxWgt_)
        continue;
      //---

      if (pv->tracksSize() < minVtxTracks_)
        continue;

      const auto& testTrack = pv->trackRefAt(0);
      if (testTrack.isNull() || !testTrack.isAvailable()) {
        edm::LogInfo("") << "Track collection not found. Skipping cut on sumPt.";
      } else {
        double sumPt = 0;
        for (auto iTrack = pv->tracks_begin(); iTrack != pv->tracks_end(); ++iTrack) {
          const auto pt = (*iTrack)->pt();
          sumPt += pt;
        }
        if (sumPt < minSumPt_)
          continue;
      }

      hPVx->Fill(pv->x(), pv->z());
      hPVy->Fill(pv->y(), pv->z());

      //
      // 3D fit section
      //
      // apply additional quality cut
      if (pvQuality(*pv) > dynamicQualityCut_)
        continue;
      // if store exceeds max. size: reduce size and apply new quality cut
      if (pvStore_.size() >= maxNrVertices_) {
        compressStore();
        if (pvQuality(*pv) > dynamicQualityCut_)
          continue;
      }
      //
      // copy PV to store
      //
      int bx = iEvent.bunchCrossing();
      BeamSpotFitPVData pvData;
      pvData.bunchCrossing = bx;
      pvData.position[0] = pv->x();
      pvData.position[1] = pv->y();
      pvData.position[2] = pv->z();
      pvData.posError[0] = pv->xError();
      pvData.posError[1] = pv->yError();
      pvData.posError[2] = pv->zError();
      pvData.posCorr[0] = pv->covariance(0, 1) / pv->xError() / pv->yError();
      pvData.posCorr[1] = pv->covariance(0, 2) / pv->xError() / pv->zError();
      pvData.posCorr[2] = pv->covariance(1, 2) / pv->yError() / pv->zError();
      pvStore_.push_back(pvData);

      if (ftree_ != nullptr) {
        theBeamSpotTreeData_.run(iEvent.id().run());
        theBeamSpotTreeData_.lumi(iEvent.luminosityBlock());
        theBeamSpotTreeData_.bunchCrossing(bx);
        theBeamSpotTreeData_.pvData(pvData);
        ftree_->Fill();
      }

      if (fFitPerBunchCrossing)
        bxMap_[bx].push_back(pvData);
    }
  }
}

void PVFitter::setTree(TTree* tree) {
  ftree_ = tree;
  theBeamSpotTreeData_.branch(ftree_);
}

bool PVFitter::runBXFitter() {
  using namespace ROOT::Minuit2;
  edm::LogInfo("PVFitter") << " Number of bunch crossings: " << bxMap_.size() << std::endl;

  bool fit_ok = true;

  for (std::map<int, std::vector<BeamSpotFitPVData> >::const_iterator pvStore = bxMap_.begin(); pvStore != bxMap_.end();
       ++pvStore) {
    // first set null beam spot in case
    // fit fails
    fbspotMap[pvStore->first] = reco::BeamSpot();

    edm::LogInfo("PVFitter") << " Number of PVs collected for PVFitter: " << (pvStore->second).size()
                             << " in bx: " << pvStore->first << std::endl;

    if ((pvStore->second).size() <= minNrVertices_) {
      edm::LogWarning("PVFitter") << " not enough PVs, continue" << std::endl;
      fit_ok = false;
      continue;
    }

    edm::LogInfo("PVFitter") << "Calculating beam spot with PVs ..." << std::endl;

    //
    // LL function and fitter
    //
    FcnBeamSpotFitPV fcn(pvStore->second);
    //
    // fit parameters: positions, widths, x-y correlations, tilts in xz and yz
    //
    MnUserParameters upar;
    upar.Add("x", 0., 0.02, -10., 10.);                                                     // 0
    upar.Add("y", 0., 0.02, -10., 10.);                                                     // 1
    upar.Add("z", 0., 0.20, -30., 30.);                                                     // 2
    upar.Add("ex", 0.015, 0.01, 0., 10.);                                                   // 3
    upar.Add("corrxy", 0., 0.02, -1., 1.);                                                  // 4
    upar.Add("ey", 0.015, 0.01, 0., 10.);                                                   // 5
    upar.Add("dxdz", 0., 0.0002, -0.1, 0.1);                                                // 6
    upar.Add("dydz", 0., 0.0002, -0.1, 0.1);                                                // 7
    upar.Add("ez", 1., 0.1, 0., 30.);                                                       // 8
    upar.Add("scale", errorScale_, errorScale_ / 10., errorScale_ / 2., errorScale_ * 2.);  // 9
    MnMigrad migrad(fcn, upar);

    //
    // first iteration without correlations
    //
    upar.Fix(4);
    upar.Fix(6);
    upar.Fix(7);
    upar.Fix(9);
    FunctionMinimum ierr = migrad(0, 1.);
    if (!ierr.IsValid()) {
      edm::LogInfo("PVFitter") << "3D beam spot fit failed in 1st iteration" << std::endl;
      fit_ok = false;
      continue;
    }
    //
    // refit with harder selection on vertices
    //
    fcn.setLimits(upar.Value(0) - sigmaCut_ * upar.Value(3),
                  upar.Value(0) + sigmaCut_ * upar.Value(3),
                  upar.Value(1) - sigmaCut_ * upar.Value(5),
                  upar.Value(1) + sigmaCut_ * upar.Value(5),
                  upar.Value(2) - sigmaCut_ * upar.Value(8),
                  upar.Value(2) + sigmaCut_ * upar.Value(8));
    ierr = migrad(0, 1.);
    if (!ierr.IsValid()) {
      edm::LogInfo("PVFitter") << "3D beam spot fit failed in 2nd iteration" << std::endl;
      fit_ok = false;
      continue;
    }
    //
    // refit with correlations
    //
    upar.Release(4);
    upar.Release(6);
    upar.Release(7);
    ierr = migrad(0, 1.);
    if (!ierr.IsValid()) {
      edm::LogInfo("PVFitter") << "3D beam spot fit failed in 3rd iteration" << std::endl;
      fit_ok = false;
      continue;
    }

    fwidthX = upar.Value(3);
    fwidthY = upar.Value(5);
    fwidthZ = upar.Value(8);
    fwidthXerr = upar.Error(3);
    fwidthYerr = upar.Error(5);
    fwidthZerr = upar.Error(8);

    reco::BeamSpot::CovarianceMatrix matrix;
    // need to get the full cov matrix
    matrix(0, 0) = pow(upar.Error(0), 2);
    matrix(1, 1) = pow(upar.Error(1), 2);
    matrix(2, 2) = pow(upar.Error(2), 2);
    matrix(3, 3) = fwidthZerr * fwidthZerr;
    matrix(4, 4) = pow(upar.Error(6), 2);
    matrix(5, 5) = pow(upar.Error(7), 2);
    matrix(6, 6) = fwidthXerr * fwidthXerr;

    fbeamspot = reco::BeamSpot(reco::BeamSpot::Point(upar.Value(0), upar.Value(1), upar.Value(2)),
                               fwidthZ,
                               upar.Value(6),
                               upar.Value(7),
                               fwidthX,
                               matrix);
    fbeamspot.setBeamWidthX(fwidthX);
    fbeamspot.setBeamWidthY(fwidthY);
    fbeamspot.setType(reco::BeamSpot::Tracker);

    fbspotMap[pvStore->first] = fbeamspot;
    edm::LogInfo("PVFitter") << "3D PV fit done for this bunch crossing." << std::endl;
    fit_ok = fit_ok & true;
  }

  return fit_ok;
}

bool PVFitter::runFitter() {
  using namespace ROOT::Minuit2;
  edm::LogInfo("PVFitter") << " Number of PVs collected for PVFitter: " << pvStore_.size() << std::endl;

  if (pvStore_.size() <= minNrVertices_)
    return false;

  //need to create a unique histogram name to avoid ROOT trying to re-use one
  // also tell ROOT to hide its global state
  TDirectory::TContext guard{nullptr};
  std::ostringstream str;
  str << this;
  std::unique_ptr<TH1D> h1PVx{hPVx->ProjectionX((std::string("h1PVx") + str.str()).c_str(), 0, -1, "e")};
  std::unique_ptr<TH1D> h1PVy{hPVy->ProjectionX((std::string("h1PVy") + str.str()).c_str(), 0, -1, "e")};
  std::unique_ptr<TH1D> h1PVz{hPVx->ProjectionY((std::string("h1PVz") + str.str()).c_str(), 0, -1, "e")};

  //Use our own copy for thread safety
  // also do not add to global list of functions
  TF1 gausx("localGausX", "gaus", 0., 1., TF1::EAddToList::kNo);
  TF1 gausy("localGausY", "gaus", 0., 1., TF1::EAddToList::kNo);
  TF1 gausz("localGausZ", "gaus", 0., 1., TF1::EAddToList::kNo);

  h1PVx->Fit(&gausx, "QLMN0 SERIAL");
  h1PVy->Fit(&gausy, "QLMN0 SERIAL");
  h1PVz->Fit(&gausz, "QLMN0 SERIAL");

  fwidthX = gausx.GetParameter(2);
  fwidthY = gausy.GetParameter(2);
  fwidthZ = gausz.GetParameter(2);
  fwidthXerr = gausx.GetParError(2);
  fwidthYerr = gausy.GetParError(2);
  fwidthZerr = gausz.GetParError(2);

  double estX = gausx.GetParameter(1);
  double estY = gausy.GetParameter(1);
  double estZ = gausz.GetParameter(1);
  double errX = fwidthX * 3.;
  double errY = fwidthY * 3.;
  double errZ = fwidthZ * 3.;

  if (!do3DFit_) {
    reco::BeamSpot::CovarianceMatrix matrix;
    matrix(2, 2) = gausz.GetParError(1) * gausz.GetParError(1);
    matrix(3, 3) = fwidthZerr * fwidthZerr;
    matrix(6, 6) = fwidthXerr * fwidthXerr;

    fbeamspot =
        reco::BeamSpot(reco::BeamSpot::Point(gausx.GetParameter(1), gausy.GetParameter(1), gausz.GetParameter(1)),
                       fwidthZ,
                       0.,
                       0.,
                       fwidthX,
                       matrix);
    fbeamspot.setBeamWidthX(fwidthX);
    fbeamspot.setBeamWidthY(fwidthY);
    fbeamspot.setType(reco::BeamSpot::Tracker);

  } else {  // do 3D fit
    //
    // LL function and fitter
    //
    FcnBeamSpotFitPV fcn(pvStore_);
    //
    // fit parameters: positions, widths, x-y correlations, tilts in xz and yz
    //
    MnUserParameters upar;
    upar.Add("x", estX, errX, -10., 10.);                                                   // 0
    upar.Add("y", estY, errY, -10., 10.);                                                   // 1
    upar.Add("z", estZ, errZ, -30., 30.);                                                   // 2
    upar.Add("ex", 0.015, 0.01, 0., 10.);                                                   // 3
    upar.Add("corrxy", 0., 0.02, -1., 1.);                                                  // 4
    upar.Add("ey", 0.015, 0.01, 0., 10.);                                                   // 5
    upar.Add("dxdz", 0., 0.0002, -0.1, 0.1);                                                // 6
    upar.Add("dydz", 0., 0.0002, -0.1, 0.1);                                                // 7
    upar.Add("ez", 1., 0.1, 0., 30.);                                                       // 8
    upar.Add("scale", errorScale_, errorScale_ / 10., errorScale_ / 2., errorScale_ * 2.);  // 9
    MnMigrad migrad(fcn, upar);
    //
    // first iteration without correlations
    //
    migrad.Fix(4);
    migrad.Fix(6);
    migrad.Fix(7);
    migrad.Fix(9);
    FunctionMinimum ierr = migrad(0, 1.);
    if (!ierr.IsValid()) {
      edm::LogWarning("PVFitter") << "3D beam spot fit failed in 1st iteration" << std::endl;
      return false;
    }
    //
    // refit with harder selection on vertices
    //

    vector<double> results;
    vector<double> errors;
    results = ierr.UserParameters().Params();
    errors = ierr.UserParameters().Errors();

    fcn.setLimits(results[0] - sigmaCut_ * results[3],
                  results[0] + sigmaCut_ * results[3],
                  results[1] - sigmaCut_ * results[5],
                  results[1] + sigmaCut_ * results[5],
                  results[2] - sigmaCut_ * results[8],
                  results[2] + sigmaCut_ * results[8]);
    ierr = migrad(0, 1.);
    if (!ierr.IsValid()) {
      edm::LogWarning("PVFitter") << "3D beam spot fit failed in 2nd iteration" << std::endl;
      return false;
    }
    //
    // refit with correlations
    //
    migrad.Release(4);
    migrad.Release(6);
    migrad.Release(7);
    ierr = migrad(0, 1.);
    if (!ierr.IsValid()) {
      edm::LogWarning("PVFitter") << "3D beam spot fit failed in 3rd iteration" << std::endl;
      return false;
    }

    results = ierr.UserParameters().Params();
    errors = ierr.UserParameters().Errors();

    fwidthX = results[3];
    fwidthY = results[5];
    fwidthZ = results[8];
    fwidthXerr = errors[3];
    fwidthYerr = errors[5];
    fwidthZerr = errors[8];

    // check errors on widths and sigmaZ for nan
    if (edm::isNotFinite(fwidthXerr) || edm::isNotFinite(fwidthYerr) || edm::isNotFinite(fwidthZerr)) {
      edm::LogWarning("PVFitter") << "3D beam spot fit returns nan in 3rd iteration" << std::endl;
      return false;
    }

    reco::BeamSpot::CovarianceMatrix matrix;
    // need to get the full cov matrix
    matrix(0, 0) = pow(errors[0], 2);
    matrix(1, 1) = pow(errors[1], 2);
    matrix(2, 2) = pow(errors[2], 2);
    matrix(3, 3) = fwidthZerr * fwidthZerr;
    matrix(6, 6) = fwidthXerr * fwidthXerr;

    fbeamspot = reco::BeamSpot(
        reco::BeamSpot::Point(results[0], results[1], results[2]), fwidthZ, results[6], results[7], fwidthX, matrix);
    fbeamspot.setBeamWidthX(fwidthX);
    fbeamspot.setBeamWidthY(fwidthY);
    fbeamspot.setType(reco::BeamSpot::Tracker);
  }

  return true;
}

void PVFitter::dumpTxtFile() {}

void PVFitter::compressStore() {
  //
  // fill vertex qualities
  //
  pvQualities_.resize(pvStore_.size());
  for (unsigned int i = 0; i < pvStore_.size(); ++i)
    pvQualities_[i] = pvQuality(pvStore_[i]);
  sort(pvQualities_.begin(), pvQualities_.end());
  //
  // Set new quality cut to median. This cut will be used to reduce the
  // number of vertices in the store and also apply to all new vertices
  // until the next reset
  //
  dynamicQualityCut_ = pvQualities_[pvQualities_.size() / 2];
  //
  // remove all vertices failing the cut from the store
  //   (to be moved to a more efficient memory management!)
  //
  unsigned int iwrite(0);
  for (unsigned int i = 0; i < pvStore_.size(); ++i) {
    if (pvQuality(pvStore_[i]) > dynamicQualityCut_)
      continue;
    if (i != iwrite)
      pvStore_[iwrite] = pvStore_[i];
    ++iwrite;
  }
  pvStore_.resize(iwrite);
  edm::LogInfo("PVFitter") << "Reduced primary vertex store size to " << pvStore_.size()
                           << " ; new dynamic quality cut = " << dynamicQualityCut_ << std::endl;
}

double PVFitter::pvQuality(const reco::Vertex& pv) const {
  //
  // determinant of the transverse part of the PV covariance matrix
  //
  return pv.covariance(0, 0) * pv.covariance(1, 1) - pv.covariance(0, 1) * pv.covariance(0, 1);
}

double PVFitter::pvQuality(const BeamSpotFitPVData& pv) const {
  //
  // determinant of the transverse part of the PV covariance matrix
  //
  double ex = pv.posError[0];
  double ey = pv.posError[1];
  return ex * ex * ey * ey * (1 - pv.posCorr[0] * pv.posCorr[0]);
}
