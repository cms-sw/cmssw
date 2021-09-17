/** \class KFFittingSmoother
 *  A TrajectorySmoother that rpeats the forward fit before smoothing.
 *  This is necessary e.g. when the seed introduced a bias (by using
 *  a beam contraint etc.). Ported from ORCA
 *
 *  \author todorov, cerati
 */

#include "TrackingTools/PatternTools/interface/TrajectorySmoother.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryFitter.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "TrackingTools/TransientTrackingRecHit/interface/InvalidTransientRecHit.h"
#include "CommonTools/Utils/interface/DynArray.h"

namespace {

  struct KFFittingSmootherParam {
    explicit KFFittingSmootherParam(const edm::ParameterSet& conf)
        : theEstimateCut(conf.getParameter<double>("EstimateCut")),
          theMaxFractionOutliers(conf.getParameter<double>("MaxFractionOutliers")),
          theMaxNumberOfOutliers(conf.getParameter<int>("MaxNumberOfOutliers")),
          theNoOutliersBeginEnd(conf.getParameter<bool>("NoOutliersBeginEnd")),
          theMinDof(conf.getParameter<int>("MinDof")),
          theMinNumberOfHits(conf.getParameter<int>("MinNumberOfHits")),
          rejectTracksFlag(conf.getParameter<bool>("RejectTracks")),
          breakTrajWith2ConsecutiveMissing(conf.getParameter<bool>("BreakTrajWith2ConsecutiveMissing")),
          noInvalidHitsBeginEnd(conf.getParameter<bool>("NoInvalidHitsBeginEnd")) {}

    double theEstimateCut;

    float theMaxFractionOutliers;
    int theMaxNumberOfOutliers;
    bool theNoOutliersBeginEnd;
    int theMinDof;

    int theMinNumberOfHits;
    bool rejectTracksFlag;
    bool breakTrajWith2ConsecutiveMissing;
    bool noInvalidHitsBeginEnd;
  };

  class KFFittingSmoother final : public TrajectoryFitter, private KFFittingSmootherParam {
  public:
    ~KFFittingSmoother() override {}

    KFFittingSmoother(const TrajectoryFitter& aFitter,
                      const TrajectorySmoother& aSmoother,
                      const edm::ParameterSet& conf)
        : KFFittingSmootherParam(conf), theFitter(aFitter.clone()), theSmoother(aSmoother.clone()) {}

  private:
    static void fillDescriptions(edm::ParameterSetDescription& desc) {
      desc.add<double>("EstimateCut", -1);
      desc.add<double>("MaxFractionOutliers", 0.3);
      desc.add<int>("MaxNumberOfOutliers", 3);
      desc.add<int>("MinDof", 2);
      desc.add<bool>("NoOutliersBeginEnd", false);
      desc.add<int>("MinNumberOfHits", 5);
      desc.add<bool>("RejectTracks", true);
      desc.add<bool>("BreakTrajWith2ConsecutiveMissing", true);
      desc.add<bool>("NoInvalidHitsBeginEnd", true);
      desc.add<double>("LogPixelProbabilityCut", 0);
    }

    Trajectory fitOne(const Trajectory& t, fitType type) const override;
    Trajectory fitOne(const TrajectorySeed& aSeed,
                      const RecHitContainer& hits,
                      const TrajectoryStateOnSurface& firstPredTsos,
                      fitType type) const override;
    Trajectory fitOne(const TrajectorySeed& aSeed, const RecHitContainer& hits, fitType type) const override;

    const TrajectoryFitter* fitter() const { return theFitter.get(); }
    const TrajectorySmoother* smoother() const { return theSmoother.get(); }

    std::unique_ptr<TrajectoryFitter> clone() const override {
      return std::unique_ptr<TrajectoryFitter>(new KFFittingSmoother(*theFitter, *theSmoother, *this));
    }

    void setHitCloner(TkCloner const* hc) override {
      theFitter->setHitCloner(hc);
      theSmoother->setHitCloner(hc);
    }

    KFFittingSmoother(const TrajectoryFitter& aFitter,
                      const TrajectorySmoother& aSmoother,
                      KFFittingSmootherParam const& other)
        : KFFittingSmootherParam(other), theFitter(aFitter.clone()), theSmoother(aSmoother.clone()) {}

    Trajectory smoothingStep(Trajectory&& fitted) const {
      if (theEstimateCut > 0) {
        // remove "outlier" at the end of Traj
        while (
            !fitted.empty() && fitted.foundHits() >= theMinNumberOfHits &&
            (!fitted.lastMeasurement().recHitR().isValid() || (fitted.lastMeasurement().recHitR().det() != nullptr &&
                                                               fitted.lastMeasurement().estimate() > theEstimateCut)))
          fitted.pop();
        if (fitted.foundHits() < theMinNumberOfHits)
          return Trajectory();
      }
      return theSmoother->trajectory(fitted);
    }

  private:
    const std::unique_ptr<TrajectoryFitter> theFitter;
    const std::unique_ptr<TrajectorySmoother> theSmoother;

    /// Method to check that the trajectory has no NaN in the states and chi2
    static bool checkForNans(const Trajectory& theTraj);

    friend class KFFittingSmootherESProducer;
  };

  // #define VI_DEBUG

#ifdef VI_DEBUG
#define DPRINT(x) std::cout << x << ": "
#define PRINT std::cout
#else
#define DPRINT(x) LogTrace(x)
#define PRINT LogTrace("")
#endif

  inline Trajectory KFFittingSmoother::fitOne(const Trajectory& t, fitType type) const {
    if (!t.isValid())
      return Trajectory();
    return smoothingStep(theFitter->fitOne(t, type));
  }

  inline bool KFFittingSmoother::checkForNans(const Trajectory& theTraj) {
    if (edm::isNotFinite(theTraj.chiSquared()))
      return false;
    auto const& vtm = theTraj.measurements();
    for (auto const& tm : vtm) {
      if (edm::isNotFinite(tm.estimate()))
        return false;
      auto const& v = tm.updatedState().localParameters().vector();
      for (int i = 0; i < 5; ++i)
        if (edm::isNotFinite(v[i]))
          return false;
      if (!tm.updatedState().curvilinearError().posDef())
        return false;  //not a NaN by itself, but will lead to one
      auto const& m = tm.updatedState().curvilinearError().matrix();
      for (int i = 0; i < 5; ++i)
        for (int j = 0; j < (i + 1); ++j)
          if (edm::isNotFinite(m(i, j)))
            return false;
    }
    return true;
  }

  namespace {
    inline void print(const std::string& header, const TrajectoryStateOnSurface& tsos) {
      DPRINT("TrackFitters") << header << tsos.globalPosition().perp() << ' ' << tsos.globalPosition().z() << ' '
                             << 1. / tsos.signedInverseMomentum() << ' ' << 1. / tsos.transverseCurvature() << ' '
                             << tsos.globalMomentum().eta() << std::endl;
    }
  }  // namespace

  inline Trajectory KFFittingSmoother::fitOne(const TrajectorySeed& aSeed,
                                              const RecHitContainer& hits,
                                              const TrajectoryStateOnSurface& firstPredTsos,
                                              fitType type) const {
    LogDebug("TrackFitters") << "In KFFittingSmoother::fit";

    print("firstPred ", firstPredTsos);

    if (hits.empty())
      return Trajectory();

    RecHitContainer myHits = hits;
    Trajectory tmp_first;

    //call the fitter
    Trajectory smoothed = smoothingStep(theFitter->fitOne(aSeed, myHits, firstPredTsos));

    do {
#ifdef EDM_ML_DEBUG
      //if no outliers the fit is done only once
      for (unsigned int j = 0; j < myHits.size(); j++) {
        if (myHits[j]->det())
          LogTrace("TrackFitters") << "hit #:" << j + 1 << " rawId=" << myHits[j]->det()->geographicalId().rawId()
                                   << " validity=" << myHits[j]->isValid();
        else
          LogTrace("TrackFitters") << "hit #:" << j + 1 << " Hit with no Det information";
      }
#endif

#if defined(VI_DEBUG) || defined(EDM_ML_DEBUG)
      if (smoothed.isValid()) {
        print("first state ", smoothed.firstMeasurement().updatedState());
        print("last  state ", smoothed.lastMeasurement().updatedState());
      }
#endif

      bool hasNaN = false;
      if (!smoothed.isValid() || (hasNaN = !checkForNans(smoothed)) || (smoothed.foundHits() < theMinNumberOfHits)) {
        if (hasNaN)
          edm::LogWarning("TrackNaN") << "Track has NaN or the cov is not pos-definite";
        if (smoothed.foundHits() < theMinNumberOfHits)
          LogTrace("TrackFitters") << "smoothed.foundHits()<theMinNumberOfHits";
        DPRINT("TrackFitters") << "smoothed invalid => trajectory rejected with nhits/chi2 " << smoothed.foundHits()
                               << '/' << smoothed.chiSquared() << "\n";
        if (rejectTracksFlag) {
          return Trajectory();
        } else {
          std::swap(smoothed, tmp_first);  // if first attempt, tmp_first would be invalid anyway
          DPRINT("TrackFitters") << "smoothed invalid => returning orignal trajectory with nhits/chi2 "
                                 << smoothed.foundHits() << '/' << smoothed.chiSquared() << "\n";
        }
        break;
      }
#ifdef EDM_ML_DEBUG
      else {
        LogTrace("TrackFitters") << "dump hits after smoothing";
        Trajectory::DataContainer meas = smoothed.measurements();
        for (Trajectory::DataContainer::iterator it = meas.begin(); it != meas.end(); ++it) {
          LogTrace("TrackFitters") << "hit #" << meas.end() - it - 1 << " validity=" << it->recHit()->isValid()
                                   << " det=" << it->recHit()->geographicalId().rawId();
        }
      }
#endif

      if (myHits.size() != smoothed.measurements().size())
        DPRINT("TrackFitters") << "lost hits. before/after: " << myHits.size() << '/' << smoothed.measurements().size()
                               << "\n";

      if (theEstimateCut <= 0)
        break;

      // Check if there are outliers

      auto msize = smoothed.measurements().size();
      declareDynArray(unsigned int, msize, bad);
      unsigned int nbad = 0;
      unsigned int ind = 0;
      unsigned int lastValid = smoothed.measurements().size();
      for (auto const& tm : smoothed.measurements()) {
        if (tm.estimate() > theEstimateCut &&
            tm.recHitR().det() != nullptr  // do not consider outliers constraints and other special "hits"
        )
          bad[nbad++] = ind;
        if (ind < lastValid && tm.recHitR().det() != nullptr && tm.recHitR().isValid())
          lastValid = ind;
        ++ind;
      }

      if (0 == nbad)
        break;

      DPRINT("TrackFitters") << "size/found/outliers list " << smoothed.measurements().size() << '/'
                             << smoothed.foundHits() << ' ' << nbad << ": ";
      for (auto i = 0U; i < nbad; ++i)
        PRINT << bad[i] << ',';
      PRINT << std::endl;

      if (
          //	 (smoothed.foundHits() == theMinNumberOfHits)  ||
          int(nbad) > theMaxNumberOfOutliers || float(nbad) > theMaxFractionOutliers * float(smoothed.foundHits())) {
        DPRINT("TrackFitters") << "smoothed low quality => trajectory with nhits/chi2 " << smoothed.foundHits() << '/'
                               << smoothed.chiSquared() << "\n";
        PRINT << "try to remove " << lastValid << std::endl;
        nbad = 0;  // try to short the traj...  (below lastValid will be added)

        // do not perform outliers rejection if track is already low quality
        /*
      if ( rejectTracksFlag  && (smoothed.chiSquared() > theEstimateCut*smoothed.ndof())  ) {
	DPRINT("TrackFitters") << "smoothed low quality => trajectory rejected with nhits/chi2 " << smoothed.foundHits() << '/' <<  smoothed.chiSquared() << "\n";
        return Trajectory();
      } else {
	DPRINT("TrackFitters") << "smoothed low quality => return original trajectory with nhits/chi2 " << smoothed.foundHits() << '/' <<  smoothed.chiSquared() << "\n";
      }
      break;
      */
      }

      // always add last valid hit  as outlier candidate
      bad[nbad++] = lastValid;

      // if ( (smoothed.ndof()<theMinDof) |  ) break;

      assert(smoothed.measurements().size() <= myHits.size());

      myHits.resize(smoothed.measurements().size());  // hits are only removed from the back...

      assert(smoothed.measurements().size() == myHits.size());

      declareDynArray(Trajectory, nbad, smoothedCand);

      auto NHits = myHits.size();
      float minChi2 = std::numeric_limits<float>::max();

      auto loc = nbad;
      for (auto i = 0U; i < nbad; ++i) {
        auto j = NHits - bad[i] - 1;
        assert(myHits[j]->geographicalId() == smoothed.measurements()[bad[i]].recHitR().geographicalId());
        auto removedHit = myHits[j];
        myHits[j] = std::make_shared<InvalidTrackingRecHit>(*removedHit->det(), TrackingRecHit::missing);
        smoothedCand[i] = smoothingStep(theFitter->fitOne(aSeed, myHits, firstPredTsos));
        myHits[j] = removedHit;
        if (smoothedCand[i].isValid() && smoothedCand[i].chiSquared() < minChi2) {
          minChi2 = smoothedCand[i].chiSquared();
          loc = i;
        }
      }

      if (loc == nbad) {
        DPRINT("TrackFitters") << "New trajectories all invalid"
                               << "\n";
        return Trajectory();
      }

      DPRINT("TrackFitters") << "outlier removed " << bad[loc] << '/' << minChi2 << " was " << smoothed.chiSquared()
                             << "\n";

      if (minChi2 > smoothed.chiSquared()) {
        DPRINT("TrackFitters") << "removing outlier makes chi2 worse " << minChi2 << '/' << smoothed.chiSquared()
                               << "\nOri: ";
        for (auto const& tm : smoothed.measurements())
          PRINT << tm.recHitR().geographicalId() << '/' << tm.estimate() << ' ';
        PRINT << "\nNew: ";
        for (auto const& tm : smoothedCand[loc].measurements())
          PRINT << tm.recHitR().geographicalId() << '/' << tm.estimate() << ' ';
        PRINT << "\n";

        // return Trajectory();
        // break;
      }

      std::swap(smoothed, tmp_first);
      myHits[NHits - bad[loc] - 1] =
          std::make_shared<InvalidTrackingRecHit>(*myHits[NHits - bad[loc] - 1]->det(), TrackingRecHit::missing);
      std::swap(smoothed, smoothedCand[loc]);
      // firstTry=false;

      DPRINT("TrackFitters") << "new trajectory with nhits/chi2 " << smoothed.foundHits() << '/'
                             << smoothed.chiSquared() << "\n";

      // Look if there are two consecutive invalid hits  FIXME:  take into account split matched hits!!!
      if (breakTrajWith2ConsecutiveMissing) {
        unsigned int firstinvalid = myHits.size();
        for (unsigned int j = 0; j < myHits.size() - 1; ++j) {
          if (((myHits[j]->type() == TrackingRecHit::missing) && (myHits[j]->geographicalId().rawId() != 0)) &&
              ((myHits[j + 1]->type() == TrackingRecHit::missing) && (myHits[j + 1]->geographicalId().rawId() != 0)) &&
              ((myHits[j]->geographicalId().rawId() & (~3)) !=
               (myHits[j + 1]->geographicalId().rawId() & (~3)))  // same gluedDet
          ) {
            firstinvalid = j;
            DPRINT("TrackFitters") << "Found two consecutive missing hits. First invalid: " << firstinvalid << "\n";
            break;
          }
        }

        //reject all the hits after the last valid before two consecutive invalid (missing) hits
        //hits are sorted in the same order as in the track candidate FIXME??????
        if (firstinvalid != myHits.size()) {
          myHits.erase(myHits.begin() + firstinvalid, myHits.end());
          smoothed = smoothingStep(theFitter->fitOne(aSeed, myHits, firstPredTsos));
          DPRINT("TrackFitters") << "Trajectory shortened " << smoothed.foundHits() << '/' << smoothed.chiSquared()
                                 << "\n";
        }
      }

    }  // do
    while (true);

    if (smoothed.isValid()) {
      if (noInvalidHitsBeginEnd && !smoothed.empty()  //should we send a warning ?
      ) {
        // discard latest dummy measurements
        if (!smoothed.empty() && !smoothed.lastMeasurement().recHitR().isValid())
          LogTrace("TrackFitters") << "Last measurement is invalid";

        while (!smoothed.empty() && !smoothed.lastMeasurement().recHitR().isValid())
          smoothed.pop();

        //remove the invalid hits at the begin of the trajectory
        if (!smoothed.empty() && !smoothed.firstMeasurement().recHitR().isValid()) {
          LogTrace("TrackFitters") << "First measurement is in`valid";
          Trajectory tmpTraj(smoothed.seed(), smoothed.direction());
          Trajectory::DataContainer& meas = smoothed.measurements();
          auto it = meas.begin();
          for (; it != meas.end(); ++it)
            if (it->recHitR().isValid())
              break;
          //push the first valid measurement and set the same global chi2
          tmpTraj.push(std::move(*it), smoothed.chiSquared());

          for (auto itt = it + 1; itt != meas.end(); ++itt)
            tmpTraj.push(std::move(*itt), 0);  //add all the other measurements

          std::swap(smoothed, tmpTraj);

        }  //  if ( !smoothed.firstMeasurement().recHit()->isValid() )

      }  // if ( noInvalidHitsBeginEnd )

      LogTrace("TrackFitters") << "end: returning smoothed trajectory with chi2=" << smoothed.chiSquared();

      //LogTrace("TrackFitters") << "dump hits before return";
      //Trajectory::DataContainer meas = smoothed.measurements();
      //for (Trajectory::DataContainer::iterator it=meas.begin();it!=meas.end();++it) {
      //LogTrace("TrackFitters") << "hit #" << meas.end()-it-1 << " validity=" << it->recHit()->isValid()
      //<< " det=" << it->recHit()->geographicalId().rawId();
      //}
    }

    return smoothed;
  }

  inline Trajectory KFFittingSmoother::fitOne(const TrajectorySeed& aSeed,
                                              const RecHitContainer& hits,
                                              fitType type) const {
    throw cms::Exception("TrackFitters",
                         "KFFittingSmoother::fit(TrajectorySeed, <TransientTrackingRecHit>) not implemented");

    return Trajectory();
  }

}  // namespace

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/ESProducer.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryFitterRecord.h"

namespace {

  class KFFittingSmootherESProducer final : public edm::ESProducer {
  public:
    KFFittingSmootherESProducer(const edm::ParameterSet& p) : pset_{p} {
      std::string myname = p.getParameter<std::string>("ComponentName");
      auto cc = setWhatProduced(this, myname);
      fitToken_ = cc.consumes(edm::ESInputTag("", pset_.getParameter<std::string>("Fitter")));
      smoothToken_ = cc.consumes(edm::ESInputTag("", pset_.getParameter<std::string>("Smoother")));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<std::string>("ComponentName", "KFFittingSmoother");
      desc.add<std::string>("Fitter", "KFFitter");
      desc.add<std::string>("Smoother", "KFSmoother");
      KFFittingSmoother::fillDescriptions(desc);
      descriptions.add("KFFittingSmoother", desc);
    }

    std::unique_ptr<TrajectoryFitter> produce(const TrajectoryFitterRecord& iRecord) {
      return std::make_unique<KFFittingSmoother>(iRecord.get(fitToken_), iRecord.get(smoothToken_), pset_);
    }

  private:
    const edm::ParameterSet pset_;
    edm::ESGetToken<TrajectoryFitter, TrajectoryFitterRecord> fitToken_;
    edm::ESGetToken<TrajectorySmoother, TrajectoryFitterRecord> smoothToken_;
  };
}  // namespace

#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_EVENTSETUP_MODULE(KFFittingSmootherESProducer);
