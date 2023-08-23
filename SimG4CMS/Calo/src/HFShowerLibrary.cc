///////////////////////////////////////////////////////////////////////////////
// File: HFShowerLibrary.cc
// Description: Shower library for Very forward hadron calorimeter
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/HFShowerLibrary.h"
#include "SimDataFormats/CaloHit/interface/HFShowerLibraryEventInfo.h"
#include "SimG4Core/Notification/interface/G4TrackToParticleID.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "G4VPhysicalVolume.hh"
#include "G4NavigationHistory.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4ParticleTable.hh"
#include "Randomize.hh"
#include "CLHEP/Units/SystemOfUnits.h"
#include "CLHEP/Units/PhysicalConstants.h"

//#define EDM_ML_DEBUG
namespace {
  HFShowerLibrary::Params paramsFrom(edm::ParameterSet const& hfShower,
                                     edm::ParameterSet const& hfShowerLibrary,
                                     double iDeltaPhi) {
    HFShowerLibrary::Params params;

    params.dphi_ = iDeltaPhi;

    params.probMax_ = hfShower.getParameter<double>("ProbMax");
    params.equalizeTimeShift_ = hfShower.getParameter<bool>("EqualizeTimeShift");

    params.backProb_ = hfShowerLibrary.getParameter<double>("BackProbability");
    params.verbose_ = hfShowerLibrary.getUntrackedParameter<bool>("Verbosity", false);
    params.applyFidCut_ = hfShowerLibrary.getParameter<bool>("ApplyFiducialCut");

    return params;
  }

  HFShowerLibrary::FileParams fileParamsFrom(edm::ParameterSet const& hfShowerLibrary) {
    HFShowerLibrary::FileParams params;

    edm::FileInPath fp = hfShowerLibrary.getParameter<edm::FileInPath>("FileName");
    params.fileName_ = fp.fullPath();
    if (params.fileName_.find('.') == 0)
      params.fileName_.erase(0, 2);

    std::string emName = hfShowerLibrary.getParameter<std::string>("TreeEMID");
    std::string hadName = hfShowerLibrary.getParameter<std::string>("TreeHadID");
    std::string branchEvInfo = hfShowerLibrary.getUntrackedParameter<std::string>(
        "BranchEvt", "HFShowerLibraryEventInfos_hfshowerlib_HFShowerLibraryEventInfo");
    std::string branchPre =
        hfShowerLibrary.getUntrackedParameter<std::string>("BranchPre", "HFShowerPhotons_hfshowerlib_");
    std::string branchPost = hfShowerLibrary.getUntrackedParameter<std::string>("BranchPost", "_R.obj");
    params.fileVersion_ = hfShowerLibrary.getParameter<int>("FileVersion");

    params.emBranchName_ = branchPre + emName + branchPost;
    params.hadBranchName_ = branchPre + hadName + branchPost;

    if (not branchEvInfo.empty()) {
      params.branchEvInfo_ = branchEvInfo + branchPost;
    }

    params.cacheBranches_ = hfShowerLibrary.getUntrackedParameter<bool>("cacheBranches", false);
    return params;
  }
}  // namespace

HFShowerLibrary::HFShowerLibrary(const HcalDDDSimConstants* hcons,
                                 const HcalSimulationParameters* hps,
                                 edm::ParameterSet const& hfShower,
                                 edm::ParameterSet const& hfShowerLibrary)
    : HFShowerLibrary(paramsFrom(hfShower, hfShowerLibrary, hcons->getPhiTableHF().front()),
                      fileParamsFrom(hfShowerLibrary),
                      HFFibre::Params(hfShower.getParameter<double>("CFibre"), hcons, hps)) {}

HFShowerLibrary::HFShowerLibrary(const std::string& name,
                                 const HcalDDDSimConstants* hcons,
                                 const HcalSimulationParameters* hps,
                                 edm::ParameterSet const& p)
    : HFShowerLibrary(
          hcons,
          hps,
          p.getParameter<edm::ParameterSet>("HFShower").getParameter<edm::ParameterSet>("HFShowerBlock"),
          p.getParameter<edm::ParameterSet>("HFShowerLibrary").getParameter<edm::ParameterSet>("HFLibraryFileBlock")) {}

HFShowerLibrary::HFShowerLibrary(const Params& iParams, const FileParams& iFileParams, HFFibre::Params iFibreParams)
    : fibre_(iFibreParams),
      hf_(),
      emBranch_(),
      hadBranch_(),
      verbose_(iParams.verbose_),
      applyFidCut_(iParams.applyFidCut_),
      equalizeTimeShift_(iParams.equalizeTimeShift_),
      probMax_(iParams.probMax_),
      backProb_(iParams.backProb_),
      dphi_(iParams.dphi_),
      rMin_(iFibreParams.rTableHF_.front()),
      rMax_(iFibreParams.rTableHF_.back()),
      gpar_(iFibreParams.gParHF_) {
  std::string pTreeName = iFileParams.fileName_;

  const char* nTree = pTreeName.c_str();
  {
    //It is not safe to open a Tfile in one thread and close in another without adding the following:
    TDirectory::TContext context;
    hf_ = std::unique_ptr<TFile>(TFile::Open(nTree));
  }
  if (!hf_->IsOpen()) {
    edm::LogError("HFShower") << "HFShowerLibrary: opening " << nTree << " failed";
    throw cms::Exception("Unknown", "HFShowerLibrary") << "Opening of " << pTreeName << " fails\n";
  } else {
    edm::LogVerbatim("HFShower") << "HFShowerLibrary: opening " << nTree << " successfully";
  }

  auto fileFormat = FileFormat::kOld;
  const int fileVersion = iFileParams.fileVersion_;

  auto newForm = iFileParams.branchEvInfo_.empty();
  TTree* event(nullptr);
  if (newForm) {
    fileFormat = FileFormat::kNew;
    event = (TTree*)hf_->Get("HFSimHits");
  } else {
    event = (TTree*)hf_->Get("Events");
  }
  VersionInfo versionInfo;
  if (event) {
    TBranch* evtInfo(nullptr);
    if (!newForm) {
      std::string info = iFileParams.branchEvInfo_;
      evtInfo = event->GetBranch(info.c_str());
    }
    if (evtInfo || newForm) {
      versionInfo = loadEventInfo(evtInfo, fileVersion);
    } else {
      edm::LogError("HFShower") << "HFShowerLibrary: HFShowerLibrayEventInfo"
                                << " Branch does not exist in Event";
      throw cms::Exception("Unknown", "HFShowerLibrary") << "Event information absent\n";
    }
  } else {
    edm::LogError("HFShower") << "HFShowerLibrary: Events Tree does not "
                              << "exist";
    throw cms::Exception("Unknown", "HFShowerLibrary") << "Events tree absent\n";
  }

  edm::LogVerbatim("HFShower").log([&](auto& logger) {
    logger << "HFShowerLibrary: Library " << versionInfo.libVers_ << " ListVersion " << versionInfo.listVersion_
           << " File version " << fileVersion << " Events Total " << totEvents_ << " and " << evtPerBin_
           << " per bin\n";
    logger << "HFShowerLibrary: Energies (GeV) with " << nMomBin_ << " bins\n";
    for (int i = 0; i < nMomBin_; ++i) {
      if (i / 10 * 10 == i && i > 0) {
        logger << "\n";
      }
      logger << "  " << pmom_[i] / CLHEP::GeV;
    }
  });

  std::string nameBr = iFileParams.emBranchName_;
  auto emBranch = event->GetBranch(nameBr.c_str());
  if (verbose_)
    emBranch->Print();
  nameBr = iFileParams.hadBranchName_;
  auto hadBranch = event->GetBranch(nameBr.c_str());
  if (verbose_)
    hadBranch->Print();

  if (emBranch->GetClassName() == std::string("vector<float>")) {
    assert(fileFormat == FileFormat::kNew);
    fileFormat = FileFormat::kNewV3;
  }
  emBranch_ = BranchReader(emBranch, fileFormat, 0, iFileParams.cacheBranches_ ? totEvents_ : 0);
  size_t offset = 0;
  if ((fileFormat == FileFormat::kNewV3 && fileVersion < 3) || (fileFormat == FileFormat::kNew && fileVersion < 2)) {
    //NOTE: for this format, the hadBranch is all empty up to
    // totEvents_ (which is more like 1/2*GenEntries())
    offset = totEvents_;
  }
  hadBranch_ = BranchReader(hadBranch, fileFormat, offset, iFileParams.cacheBranches_ ? totEvents_ : 0);

  edm::LogVerbatim("HFShower") << " HFShowerLibrary:Branch " << iFileParams.emBranchName_ << " has "
                               << emBranch->GetEntries() << " entries and Branch " << iFileParams.hadBranchName_
                               << " has " << hadBranch->GetEntries()
                               << " entries\n HFShowerLibrary::No packing information - Assume x, y, z are not in "
                                  "packed form\n Maximum probability cut off "
                               << probMax_ << "  Back propagation of light probability " << backProb_
                               << " Flag for equalizing Time Shift for different eta " << equalizeTimeShift_;

  edm::LogVerbatim("HFShower") << "HFShowerLibrary: rMIN " << rMin_ / CLHEP::cm << " cm and rMax " << rMax_ / CLHEP::cm
                               << " (Half) Phi Width of wedge " << dphi_ / CLHEP::deg;
  if (iFileParams.cacheBranches_) {
    hf_.reset();
  }
}

HFShowerLibrary::~HFShowerLibrary() {
  if (hf_)
    hf_->Close();
}

std::vector<HFShowerLibrary::Hit> HFShowerLibrary::getHits(const G4Step* aStep,
                                                           bool& isKilled,
                                                           double weight,
                                                           bool onlyLong) {
  auto const preStepPoint = aStep->GetPreStepPoint();
  auto const postStepPoint = aStep->GetPostStepPoint();
  auto const track = aStep->GetTrack();
  // Get Z-direction
  auto const aParticle = track->GetDynamicParticle();
  const G4ThreeVector& momDir = aParticle->GetMomentumDirection();
  const G4ThreeVector& hitPoint = preStepPoint->GetPosition();
  int parCode = track->GetDefinition()->GetPDGEncoding();

  // VI: for ions use internally pdg code of alpha in order to keep
  // consistency with previous simulation
  if (track->GetDefinition()->IsGeneralIon()) {
    parCode = 1000020040;
  }

#ifdef EDM_ML_DEBUG
  G4String partType = track->GetDefinition()->GetParticleName();
  const G4ThreeVector localPos = preStepPoint->GetTouchable()->GetHistory()->GetTopTransform().TransformPoint(hitPoint);
  double zoff = localPos.z() + 0.5 * gpar_[1];

  edm::LogVerbatim("HFShower") << "HFShowerLibrary::getHits " << partType << " of energy "
                               << track->GetKineticEnergy() / CLHEP::GeV << " GeV weight= " << weight
                               << " onlyLong: " << onlyLong << "  dir.orts " << momDir.x() << ", " << momDir.y() << ", "
                               << momDir.z() << "  Pos x,y,z = " << hitPoint.x() << "," << hitPoint.y() << ","
                               << hitPoint.z() << " (" << zoff << ")   sphi,cphi,stheta,ctheta  = " << sin(momDir.phi())
                               << "," << cos(momDir.phi()) << ", " << sin(momDir.theta()) << "," << cos(momDir.theta());
#endif

  double tSlice = (postStepPoint->GetGlobalTime()) / CLHEP::nanosecond;

  // use kinetic energy for protons and ions
  double pin = (track->GetDefinition()->GetBaryonNumber() > 0) ? preStepPoint->GetKineticEnergy()
                                                               : preStepPoint->GetTotalEnergy();

  return fillHits(hitPoint, momDir, parCode, pin, isKilled, weight, tSlice, onlyLong);
}

std::vector<HFShowerLibrary::Hit> HFShowerLibrary::fillHits(const G4ThreeVector& hitPoint,
                                                            const G4ThreeVector& momDir,
                                                            int parCode,
                                                            double pin,
                                                            bool& ok,
                                                            double weight,
                                                            double tSlice,
                                                            bool onlyLong) {
  std::vector<HFShowerLibrary::Hit> hit;
  ok = false;
  bool isEM = G4TrackToParticleID::isGammaElectronPositron(parCode);
  // shower is built only for gamma, e+- and stable hadrons
  if (!isEM && !G4TrackToParticleID::isStableHadron(parCode)) {
    return hit;
  }
  ok = true;

  // remove low-energy component
  const double threshold = 50 * MeV;
  if (pin < threshold) {
    return hit;
  }

  double pz = momDir.z();
  double zint = hitPoint.z();

  // if particle moves from interaction point or "backwards (halo)
  bool backward = (pz * zint < 0.) ? true : false;

  double sphi = sin(momDir.phi());
  double cphi = cos(momDir.phi());
  double ctheta = cos(momDir.theta());
  double stheta = sin(momDir.theta());

  HFShowerPhotonCollection pe;
  if (isEM) {
    if (pin < pmom_[nMomBin_ - 1]) {
      pe = interpolate(0, pin);
    } else {
      pe = extrapolate(0, pin);
    }
  } else {
    if (pin < pmom_[nMomBin_ - 1]) {
      pe = interpolate(1, pin);
    } else {
      pe = extrapolate(1, pin);
    }
  }

  std::size_t nHit = 0;
  HFShowerLibrary::Hit oneHit;
#ifdef EDM_ML_DEBUG
  int i = 0;
#endif
  for (auto const& photon : pe) {
    double zv = std::abs(photon.z());  // abs local z
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HFShower") << "HFShowerLibrary: Hit " << i++ << " " << photon << " zv " << zv;
#endif
    if (zv <= gpar_[1] && photon.lambda() > 0 && (photon.z() >= 0 || (zv > gpar_[0] && (!onlyLong)))) {
      int depth = 1;
      if (onlyLong) {
      } else if (!backward) {  // fully valid only for "front" particles
        if (photon.z() < 0)
          depth = 2;                 // with "front"-simulated shower lib.
      } else {                       // for "backward" particles - almost equal
        double r = G4UniformRand();  // share between L and S fibers
        if (r > 0.5)
          depth = 2;
      }

      // Updated coordinate transformation from local
      //  back to global using two Euler angles: phi and theta
      double pex = photon.x();
      double pey = photon.y();

      double xx = pex * ctheta * cphi - pey * sphi + zv * stheta * cphi;
      double yy = pex * ctheta * sphi + pey * cphi + zv * stheta * sphi;
      double zz = -pex * stheta + zv * ctheta;

      G4ThreeVector pos = hitPoint + G4ThreeVector(xx, yy, zz);
      zv = std::abs(pos.z()) - gpar_[4] - 0.5 * gpar_[1];
      G4ThreeVector lpos = G4ThreeVector(pos.x(), pos.y(), zv);

      zv = fibre_.zShift(lpos, depth, 0);  // distance to PMT !

      double r = pos.perp();
      double p = fibre_.attLength(photon.lambda());
      double fi = pos.phi();
      if (fi < 0)
        fi += CLHEP::twopi;
      int isect = int(fi / dphi_) + 1;
      isect = (isect + 1) / 2;
      double dfi = ((isect * 2 - 1) * dphi_ - fi);
      if (dfi < 0)
        dfi = -dfi;
      double dfir = r * sin(dfi);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HFShower") << "HFShowerLibrary: Position shift " << xx << ", " << yy << ", " << zz << ": "
                                   << pos << " R " << r << " Phi " << fi << " Section " << isect << " R*Dfi " << dfir
                                   << " Dist " << zv;
#endif
      zz = std::abs(pos.z());
      double r1 = G4UniformRand();
      double r2 = G4UniformRand();
      double r3 = backward ? G4UniformRand() : -9999.;
      if (!applyFidCut_)
        dfir += gpar_[5];

#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HFShower") << "HFShowerLibrary: rLimits " << rInside(r) << " attenuation " << r1 << ":"
                                   << exp(-p * zv) << " r2 " << r2 << " r3 " << r3 << " rDfi " << gpar_[5] << " zz "
                                   << zz << " zLim " << gpar_[4] << ":" << gpar_[4] + gpar_[1] << "\n"
                                   << "  rInside(r) :" << rInside(r) << "  r1 <= exp(-p*zv) :" << (r1 <= exp(-p * zv))
                                   << "  r2 <= probMax :" << (r2 <= probMax_ * weight)
                                   << "  r3 <= backProb :" << (r3 <= backProb_)
                                   << "  dfir > gpar[5] :" << (dfir > gpar_[5])
                                   << "  zz >= gpar[4] :" << (zz >= gpar_[4])
                                   << "  zz <= gpar[4]+gpar[1] :" << (zz <= gpar_[4] + gpar_[1]);
#endif
      if (rInside(r) && r1 <= exp(-p * zv) && r2 <= probMax_ * weight && dfir > gpar_[5] && zz >= gpar_[4] &&
          zz <= gpar_[4] + gpar_[1] && r3 <= backProb_ && (depth != 2 || zz >= gpar_[4] + gpar_[0])) {
        double tdiff = (equalizeTimeShift_) ? (fibre_.tShift(lpos, depth, -1)) : (fibre_.tShift(lpos, depth, 1));
        oneHit.position = pos;
        oneHit.depth = depth;
        oneHit.time = (tSlice + (photon.t()) + tdiff);
        hit.push_back(oneHit);

#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HFShower") << "HFShowerLibrary: Final Hit " << nHit << " position " << (hit[nHit].position)
                                     << " Depth " << (hit[nHit].depth) << " Time " << tSlice << ":" << photon.t() << ":"
                                     << tdiff << ":" << (hit[nHit].time);
#endif
        ++nHit;
      }
#ifdef EDM_ML_DEBUG
      else
        edm::LogVerbatim("HFShower") << "HFShowerLibrary: REJECTED !!!";
#endif
      if (onlyLong && zz >= gpar_[4] + gpar_[0] && zz <= gpar_[4] + gpar_[1]) {
        r1 = G4UniformRand();
        r2 = G4UniformRand();
        if (rInside(r) && r1 <= exp(-p * zv) && r2 <= probMax_ && dfir > gpar_[5]) {
          double tdiff = (equalizeTimeShift_) ? (fibre_.tShift(lpos, 2, -1)) : (fibre_.tShift(lpos, 2, 1));
          oneHit.position = pos;
          oneHit.depth = 2;
          oneHit.time = (tSlice + (photon.t()) + tdiff);
          hit.push_back(oneHit);
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HFShower") << "HFShowerLibrary: Final Hit " << nHit << " position " << (hit[nHit].position)
                                       << " Depth " << (hit[nHit].depth) << " Time " << tSlice << ":" << photon.t()
                                       << ":" << tdiff << ":" << (hit[nHit].time);
#endif
          ++nHit;
        }
      }
    }
  }

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HFShower") << "HFShowerLibrary: Total Hits " << nHit << " out of " << pe.size() << " PE";
#endif
  if (nHit > pe.size() && !onlyLong) {
    edm::LogWarning("HFShower") << "HFShowerLibrary: Hit buffer " << pe.size() << " smaller than " << nHit << " Hits";
  }
  return hit;
}

bool HFShowerLibrary::rInside(double r) const { return (r >= rMin_ && r <= rMax_); }

HFShowerLibrary::BranchCache::BranchCache(HFShowerLibrary::BranchReader& iReader, size_t maxRecordsToCache) {
  auto nRecords = iReader.numberOfRecords();
  if (nRecords > maxRecordsToCache) {
    nRecords = maxRecordsToCache;
  }
  offsets_.reserve(nRecords + 1);

  //first pass is to figure out how much space will be needed
  size_t nPhotons = 0;
  for (size_t r = 0; r < nRecords; ++r) {
    auto shower = iReader.getRecord(r + 1);
    nPhotons += shower.size();
  }

  photons_.reserve(nPhotons);

  size_t offset = 0;
  for (size_t r = 0; r < nRecords; ++r) {
    offsets_.emplace_back(offset);
    auto shower = iReader.getRecord(r + 1);
    offset += shower.size();
    std::copy(shower.begin(), shower.end(), std::back_inserter(photons_));
  }
  offsets_.emplace_back(offset);
  photons_.shrink_to_fit();
}

void HFShowerLibrary::BranchReader::doCaching(size_t maxRecordsToCache) {
  cache_ = makeCache(*this, maxRecordsToCache, branch_->GetDirectory()->GetFile()->GetName(), branch_->GetName());
}

std::shared_ptr<HFShowerLibrary::BranchCache const> HFShowerLibrary::BranchReader::makeCache(
    BranchReader& iReader, size_t maxRecordsToCache, std::string const& iFileName, std::string const& iBranchName) {
  //This allows sharing of the same cached data across the different modules (e.g. the per stream instances of OscarMTProducer
  CMS_SA_ALLOW static std::mutex s_mutex;
  std::lock_guard<std::mutex> guard(s_mutex);
  CMS_SA_ALLOW static std::map<std::pair<std::string, std::string>, std::weak_ptr<BranchCache>> s_map;
  auto v = s_map[{iFileName, iBranchName}].lock();
  if (v) {
    return v;
  }
  v = std::make_shared<BranchCache>(iReader, maxRecordsToCache);
  s_map[{iFileName, iBranchName}] = v;
  return v;
}

HFShowerPhotonCollection HFShowerLibrary::BranchCache::getRecord(int iRecord) const {
  assert(iRecord > 0);
  assert(static_cast<size_t>(iRecord + 1) < offsets_.size());

  auto start = offsets_[iRecord - 1];
  auto end = offsets_[iRecord];

  return HFShowerPhotonCollection(photons_.begin() + start, photons_.begin() + end);
}

HFShowerPhotonCollection HFShowerLibrary::BranchReader::getRecordOldForm(TBranch* iBranch, int iEntry) {
  HFShowerPhotonCollection photo;
  iBranch->SetAddress(&photo);
  iBranch->GetEntry(iEntry);
  return photo;
}
HFShowerPhotonCollection HFShowerLibrary::BranchReader::getRecordNewForm(TBranch* iBranch, int iEntry) {
  HFShowerPhotonCollection photo;

  auto temp = std::make_unique<HFShowerPhotonCollection>();
  iBranch->SetAddress(&temp);
  iBranch->GetEntry(iEntry);
  photo = std::move(*temp);

  return photo;
}
HFShowerPhotonCollection HFShowerLibrary::BranchReader::getRecordNewFormV3(TBranch* iBranch, int iEntry) {
  HFShowerPhotonCollection photo;

  std::vector<float> t;
  std::vector<float>* tp = &t;
  iBranch->SetAddress(&tp);
  iBranch->GetEntry(iEntry);
  unsigned int tSize = t.size() / 5;
  photo.reserve(tSize);
  for (unsigned int i = 0; i < tSize; i++) {
    photo.emplace_back(t[i], t[1 * tSize + i], t[2 * tSize + i], t[3 * tSize + i], t[4 * tSize + i]);
  }

  return photo;
}

HFShowerPhotonCollection HFShowerLibrary::BranchReader::getRecord(int record) const {
  if (cache_) {
    return cache_->getRecord(record);
  }
  int nrc = record - 1;
  HFShowerPhotonCollection photo;

  switch (format_) {
    case FileFormat::kNew: {
      photo = getRecordNewForm(branch_, nrc + offset_);
      break;
    }
    case FileFormat::kNewV3: {
      photo = getRecordNewFormV3(branch_, nrc + offset_);
      break;
    }
    case FileFormat::kOld: {
      photo = getRecordOldForm(branch_, nrc);
      break;
    }
  }
  return photo;
}

size_t HFShowerLibrary::BranchReader::numberOfRecords() const { return branch_->GetEntries() - offset_; }

HFShowerPhotonCollection HFShowerLibrary::getRecord(int type, int record) const {
  HFShowerPhotonCollection photo;
  if (type > 0) {
    photo = hadBranch_.getRecord(record);
  } else {
    photo = emBranch_.getRecord(record);
  }
#ifdef EDM_ML_DEBUG
  int nPhoton = photo.size();
  edm::LogVerbatim("HFShower") << "HFShowerLibrary::getRecord: Record " << record << " of type " << type << " with "
                               << nPhoton << " photons";
  for (int j = 0; j < nPhoton; j++)
    edm::LogVerbatim("HFShower") << "Photon " << j << " " << photo[j];
#endif
  return photo;
}

HFShowerLibrary::VersionInfo HFShowerLibrary::loadEventInfo(TBranch* branch, int fileVersion) {
  VersionInfo versionInfo;
  if (branch) {
    std::vector<HFShowerLibraryEventInfo> eventInfoCollection;
    branch->SetAddress(&eventInfoCollection);
    branch->GetEntry(0);
    edm::LogVerbatim("HFShower") << "HFShowerLibrary::loadEventInfo loads EventInfo Collection of size "
                                 << eventInfoCollection.size() << " records";

    totEvents_ = eventInfoCollection[0].totalEvents();
    nMomBin_ = eventInfoCollection[0].numberOfBins();
    evtPerBin_ = eventInfoCollection[0].eventsPerBin();
    versionInfo.libVers_ = eventInfoCollection[0].showerLibraryVersion();
    versionInfo.listVersion_ = eventInfoCollection[0].physListVersion();
    pmom_ = eventInfoCollection[0].energyBins();
  } else {
    edm::LogVerbatim("HFShower") << "HFShowerLibrary::loadEventInfo loads EventInfo from hardwired"
                                 << " numbers";

    nMomBin_ = 16;
    evtPerBin_ = (fileVersion == 0) ? 5000 : 10000;
    totEvents_ = nMomBin_ * evtPerBin_;
    versionInfo.libVers_ = (fileVersion == 0) ? 1.1 : 1.2;
    versionInfo.listVersion_ = 3.6;
    pmom_ = {2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 250, 350, 500, 1000};
  }
  for (int i = 0; i < nMomBin_; i++)
    pmom_[i] *= CLHEP::GeV;
  return versionInfo;
}

HFShowerPhotonCollection HFShowerLibrary::interpolate(int type, double pin) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HFShower") << "HFShowerLibrary:: Interpolate for Energy " << pin / CLHEP::GeV << " GeV with "
                               << nMomBin_ << " momentum bins and " << evtPerBin_ << " entries/bin -- total "
                               << totEvents_;
#endif
  int irc[2] = {0, 0};
  double w = 0.;
  double r = G4UniformRand();

  if (pin < pmom_[0]) {
    w = pin / pmom_[0];
    irc[1] = int(evtPerBin_ * r) + 1;
    irc[0] = 0;
  } else {
    for (int j = 0; j < nMomBin_ - 1; j++) {
      if (pin >= pmom_[j] && pin < pmom_[j + 1]) {
        w = (pin - pmom_[j]) / (pmom_[j + 1] - pmom_[j]);
        if (j == nMomBin_ - 2) {
          irc[1] = int(evtPerBin_ * 0.5 * r);
        } else {
          irc[1] = int(evtPerBin_ * r);
        }
        irc[1] += (j + 1) * evtPerBin_ + 1;
        r = G4UniformRand();
        irc[0] = int(evtPerBin_ * r) + 1 + j * evtPerBin_;
        if (irc[0] < 0) {
          edm::LogWarning("HFShower") << "HFShowerLibrary:: Illegal irc[0] = " << irc[0] << " now set to 0";
          irc[0] = 0;
        } else if (irc[0] > totEvents_) {
          edm::LogWarning("HFShower") << "HFShowerLibrary:: Illegal irc[0] = " << irc[0] << " now set to "
                                      << totEvents_;
          irc[0] = totEvents_;
        }
      }
    }
  }
  if (irc[1] < 1) {
    edm::LogWarning("HFShower") << "HFShowerLibrary:: Illegal irc[1] = " << irc[1] << " now set to 1";
    irc[1] = 1;
  } else if (irc[1] > totEvents_) {
    edm::LogWarning("HFShower") << "HFShowerLibrary:: Illegal irc[1] = " << irc[1] << " now set to " << totEvents_;
    irc[1] = totEvents_;
  }

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HFShower") << "HFShowerLibrary:: Select records " << irc[0] << " and " << irc[1] << " with weights "
                               << 1 - w << " and " << w;
#endif
  HFShowerPhotonCollection pe;

  std::size_t npold = 0;
  for (int ir = 0; ir < 2; ir++) {
    if (irc[ir] > 0) {
      auto photons = getRecord(type, irc[ir]);
      int nPhoton = photons.size();
      npold += nPhoton;
      pe.reserve(pe.size() + nPhoton);
      for (auto const& photon : photons) {
        r = G4UniformRand();
        if ((ir == 0 && r > w) || (ir > 0 && r < w)) {
          storePhoton(photon, pe);
        }
      }
    }
  }

  if ((pe.size() > npold || (npold == 0 && irc[0] > 0)) && !(pe.empty() && npold == 0))
    edm::LogWarning("HFShower") << "HFShowerLibrary: Interpolation Warning =="
                                << " records " << irc[0] << " and " << irc[1] << " gives a buffer of " << npold
                                << " photons and fills " << pe.size() << " *****";
#ifdef EDM_ML_DEBUG
  else
    edm::LogVerbatim("HFShower") << "HFShowerLibrary: Interpolation == records " << irc[0] << " and " << irc[1]
                                 << " gives a buffer of " << npold << " photons and fills " << pe.size() << " PE";
  for (std::size_t j = 0; j < pe.size(); j++)
    edm::LogVerbatim("HFShower") << "Photon " << j << " " << pe[j];
#endif
  return pe;
}

HFShowerPhotonCollection HFShowerLibrary::extrapolate(int type, double pin) {
  int nrec = int(pin / pmom_[nMomBin_ - 1]);
  double w = (pin - pmom_[nMomBin_ - 1] * nrec) / pmom_[nMomBin_ - 1];
  nrec++;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HFShower") << "HFShowerLibrary:: Extrapolate for Energy " << pin / CLHEP::GeV << " GeV with "
                               << nMomBin_ << " momentum bins and " << evtPerBin_ << " entries/bin -- "
                               << "total " << totEvents_ << " using " << nrec << " records";
#endif
  std::vector<int> irc(nrec);

  for (int ir = 0; ir < nrec; ir++) {
    double r = G4UniformRand();
    irc[ir] = int(evtPerBin_ * 0.5 * r) + (nMomBin_ - 1) * evtPerBin_ + 1;
    if (irc[ir] < 1) {
      edm::LogWarning("HFShower") << "HFShowerLibrary:: Illegal irc[" << ir << "] = " << irc[ir] << " now set to 1";
      irc[ir] = 1;
    } else if (irc[ir] > totEvents_) {
      edm::LogWarning("HFShower") << "HFShowerLibrary:: Illegal irc[" << ir << "] = " << irc[ir] << " now set to "
                                  << totEvents_;
      irc[ir] = totEvents_;
#ifdef EDM_ML_DEBUG
    } else {
      edm::LogVerbatim("HFShower") << "HFShowerLibrary::Extrapolation use irc[" << ir << "] = " << irc[ir];
#endif
    }
  }

  HFShowerPhotonCollection pe;
  std::size_t npold = 0;
  for (int ir = 0; ir < nrec; ir++) {
    if (irc[ir] > 0) {
      auto const photons = getRecord(type, irc[ir]);
      int nPhoton = photons.size();
      npold += nPhoton;
      pe.reserve(pe.size() + nPhoton);
      for (auto const& photon : photons) {
        double r = G4UniformRand();
        if (ir != nrec - 1 || r < w) {
          storePhoton(photon, pe);
        }
      }
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HFShower") << "HFShowerLibrary: Record [" << ir << "] = " << irc[ir] << " npold = " << npold;
#endif
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HFShower") << "HFShowerLibrary:: uses " << npold << " photons";
#endif

  if (pe.size() > npold || npold == 0)
    edm::LogWarning("HFShower") << "HFShowerLibrary: Extrapolation Warning == " << nrec << " records " << irc[0] << ", "
                                << irc[1] << ", ... gives a buffer of " << npold << " photons and fills " << pe.size()
                                << " *****";
#ifdef EDM_ML_DEBUG
  else
    edm::LogVerbatim("HFShower") << "HFShowerLibrary: Extrapolation == " << nrec << " records " << irc[0] << ", "
                                 << irc[1] << ", ... gives a buffer of " << npold << " photons and fills " << pe.size()
                                 << " PE";
  for (std::size_t j = 0; j < pe.size(); j++)
    edm::LogVerbatim("HFShower") << "Photon " << j << " " << pe[j];
#endif
  return pe;
}

void HFShowerLibrary::storePhoton(HFShowerPhoton const& iPhoton, HFShowerPhotonCollection& iPhotons) const {
  iPhotons.push_back(iPhoton);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HFShower") << "HFShowerLibrary: storePhoton " << iPhoton << " npe " << iPhotons.size() << " "
                               << iPhotons.back();
#endif
}
