#ifndef RecoVertex_PrimaryVertexProducer_DAClusterizerInZT_vect_h
#define RecoVertex_PrimaryVertexProducer_DAClusterizerInZT_vect_h

/**\class DAClusterizerInZT_vect

 Description: separates event tracks into clusters along the beam line

	Version which auto-vectorizes with gcc 4.6 or newer

 */

#include "RecoVertex/PrimaryVertexProducer/interface/TrackClusterizerInZ.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include <vector>
#include "DataFormats/Math/interface/Error.h"
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"

#include <memory>

//#define USEVTXDT2

class DAClusterizerInZT_vect final : public TrackClusterizerInZ {
public:
  // internal data structure for tracks
  struct track_t {
    std::vector<double> zpca_vec;                  // z-coordinate at point of closest approach to the beamline
    std::vector<double> tpca_vec;                  // t-coordinate at point of closest approach to the beamline
    std::vector<double> dz2_vec;                   // square of the error of z(pca)
    std::vector<double> dt2_vec;                   // square of the error of t(pca)
    std::vector<double> sum_Z_vec;                 // track contribution to the partition function, Z
    std::vector<double> tkwt_vec;                  // track weight, close to 1.0 for most tracks
    std::vector<unsigned int> kmin;                // index of the first cluster within zrange
    std::vector<unsigned int> kmax;                // 1 + index of the last cluster within zrange
    std::vector<const reco::TransientTrack *> tt;  // a pointer to the Transient Track

    double osumtkwt;  // 1. / (sum of all track weights)

    void addItem(double new_zpca,
                 double new_tpca,
                 double new_dz2,
                 double new_dt2,
                 const reco::TransientTrack *new_tt,
                 double new_tkwt) {
      zpca_vec.push_back(new_zpca);
      tpca_vec.push_back(new_tpca);
      dz2_vec.push_back(new_dz2);
      dt2_vec.push_back(new_dt2);
      tt.push_back(new_tt);
      tkwt_vec.push_back(new_tkwt);
      sum_Z_vec.push_back(1.0);
      kmin.push_back(0);
      kmax.push_back(0);
    }

    void insertItem(unsigned int i,
                    double new_zpca,
                    double new_tpca,
                    double new_dz2,
                    double new_dt2,
                    const reco::TransientTrack *new_tt,
                    double new_tkwt) {
      zpca_vec.insert(zpca_vec.begin() + i, new_zpca);
      tpca_vec.insert(tpca_vec.begin() + i, new_tpca);
      dz2_vec.insert(dz2_vec.begin() + i, new_dz2);
      dt2_vec.insert(dt2_vec.begin() + i, new_dt2);
      tt.insert(tt.begin() + i, new_tt);
      tkwt_vec.insert(tkwt_vec.begin() + i, new_tkwt);
      sum_Z_vec.insert(sum_Z_vec.begin() + i, 1.0);
      kmin.insert(kmin.begin() + i, 0);
      kmax.insert(kmax.begin() + i, 0);
    }

    unsigned int getSize() const { return zpca_vec.size(); }

    // has to be called everytime the items are modified
    void extractRaw() {
      zpca = &zpca_vec.front();
      tpca = &tpca_vec.front();
      dz2 = &dz2_vec.front();
      dt2 = &dt2_vec.front();
      tkwt = &tkwt_vec.front();
      sum_Z = &sum_Z_vec.front();
    }

    // pointers to the first element of vectors, needed for vectorized code
    double *__restrict__ zpca;
    double *__restrict__ tpca;
    double *__restrict__ dz2;
    double *__restrict__ dt2;
    double *__restrict__ tkwt;
    double *__restrict__ sum_Z;
  };

  // internal data structure for clusters
  struct vertex_t {
    std::vector<double> zvtx_vec;  // z coordinate
    std::vector<double> tvtx_vec;  // t coordinate
    std::vector<double> rho_vec;   //  vertex "mass" for mass-constrained clustering
#ifdef USEVTXDT2
    std::vector<double> dt2_vec;   // only used with vertex time uncertainties
    std::vector<double> sumw_vec;  // only used with vertex time uncertainties
#endif
    // --- temporary numbers, used during update
    std::vector<double> exp_arg_vec;
    std::vector<double> exp_vec;
    std::vector<double> sw_vec;
    std::vector<double> swz_vec;
    std::vector<double> swt_vec;
    std::vector<double> se_vec;
    std::vector<double> nuz_vec;
    std::vector<double> nut_vec;
    std::vector<double> szz_vec;
    std::vector<double> stt_vec;
    std::vector<double> szt_vec;

    unsigned int getSize() const { return zvtx_vec.size(); }

    void addItem(double new_zvtx, double new_tvtx, double new_rho) {
      zvtx_vec.push_back(new_zvtx);
      tvtx_vec.push_back(new_tvtx);
      rho_vec.push_back(new_rho);
      exp_arg_vec.push_back(0.0);
      exp_vec.push_back(0.0);
      swz_vec.push_back(0.0);
      swt_vec.push_back(0.0);
      se_vec.push_back(0.0);
      nuz_vec.push_back(0.0);
      nut_vec.push_back(0.0);
      szz_vec.push_back(0.0);
      stt_vec.push_back(0.0);
      szt_vec.push_back(0.0);
#ifdef USEVTXDT2
      dt2_vec.push_back(0.0);
      sumw_vec.push_back(0.0);
#endif

      extractRaw();
    }

    void insertItem(unsigned int k, double new_zvtx, double new_tvtx, double new_rho, track_t &tks) {
      zvtx_vec.insert(zvtx_vec.begin() + k, new_zvtx);
      tvtx_vec.insert(tvtx_vec.begin() + k, new_tvtx);
      rho_vec.insert(rho_vec.begin() + k, new_rho);
      exp_arg_vec.insert(exp_arg_vec.begin() + k, 0.0);
      exp_vec.insert(exp_vec.begin() + k, 0.0);
      swz_vec.insert(swz_vec.begin() + k, 0.0);
      swt_vec.insert(swt_vec.begin() + k, 0.0);
      se_vec.insert(se_vec.begin() + k, 0.0);
      nuz_vec.insert(nuz_vec.begin() + k, 0.0);
      nut_vec.insert(nut_vec.begin() + k, 0.0);
      szz_vec.insert(szz_vec.begin() + k, 0.0);
      stt_vec.insert(stt_vec.begin() + k, 0.0);
      szt_vec.insert(szt_vec.begin() + k, 0.0);
#ifdef USEVTXDT2
      dt2_vec.insert(dt2_vec.begin() + k, 0.0);
      sumw_vec.insert(sumw_vec.begin() + k, 0.0);
#endif

      // adjust vertex lists of tracks
      for (unsigned int i = 0; i < tks.getSize(); i++) {
        if (tks.kmin[i] > k) {
          tks.kmin[i]++;
        }
        if ((tks.kmax[i] >= k) || (tks.kmax[i] == tks.kmin[i])) {
          tks.kmax[i]++;
        }
      }

      extractRaw();
    }

    void removeItem(unsigned int k, track_t &tks) {
      zvtx_vec.erase(zvtx_vec.begin() + k);
      tvtx_vec.erase(tvtx_vec.begin() + k);
      rho_vec.erase(rho_vec.begin() + k);
      exp_arg_vec.erase(exp_arg_vec.begin() + k);
      exp_vec.erase(exp_vec.begin() + k);
      swz_vec.erase(swz_vec.begin() + k);
      swt_vec.erase(swt_vec.begin() + k);
      se_vec.erase(se_vec.begin() + k);
      nuz_vec.erase(nuz_vec.begin() + k);
      nut_vec.erase(nut_vec.begin() + k);
      szz_vec.erase(szz_vec.begin() + k);
      stt_vec.erase(stt_vec.begin() + k);
      szt_vec.erase(szt_vec.begin() + k);
#ifdef USEVTXDT2
      dt2_vec.erase(dt2_vec.begin() + k);
      sumw_vec.erase(sumw_vec.begin() + k);
#endif

      // adjust vertex lists of tracks
      for (unsigned int i = 0; i < tks.getSize(); i++) {
        if (tks.kmax[i] > k) {
          tks.kmax[i]--;
        }
        if ((tks.kmin[i] > k) || (((tks.kmax[i] < (tks.kmin[i] + 1)) && (tks.kmin[i] > 0)))) {
          tks.kmin[i]--;
        }
      }

      extractRaw();
    }

    unsigned int insertOrdered(double zvtx, double tvtx, double rho, track_t &tks) {
      // insert a new cluster according to it's z-position, return the index at which it was inserted

      unsigned int k = 0;
      for (; k < getSize(); k++) {
        if (zvtx < zvtx_vec[k])
          break;
      }
      insertItem(k, zvtx, tvtx, rho, tks);
      return k;
    }

    // pointers to the first element of vectors, needed for vectorized code
    double *__restrict__ zvtx;
    double *__restrict__ tvtx;
    double *__restrict__ rho;
    double *__restrict__ exp_arg;
    double *__restrict__ exp;
    double *__restrict__ swt;
    double *__restrict__ swz;
    double *__restrict__ se;
    double *__restrict__ nuz;
    double *__restrict__ nut;
    double *__restrict__ szz;
    double *__restrict__ stt;
    double *__restrict__ szt;
#ifdef USEVTXDT2
    double *__restrict__ dt2;
    double *__restrict__ sumw;
#endif

    // has to be called everytime the items are modified
    void extractRaw() {
      zvtx = &zvtx_vec.front();
      tvtx = &tvtx_vec.front();
      rho = &rho_vec.front();
      exp_arg = &exp_arg_vec.front();
      exp = &exp_vec.front();
      swz = &swz_vec.front();
      swt = &swt_vec.front();
      se = &se_vec.front();
      nuz = &nuz_vec.front();
      nut = &nut_vec.front();
      szz = &szz_vec.front();
      stt = &stt_vec.front();
      szt = &szt_vec.front();
#ifdef USEVTXDT2
      dt2 = &dt2_vec.front();
      sumw = &sumw_vec.front();
#endif
    }
  };

  DAClusterizerInZT_vect(const edm::ParameterSet &conf);

  static void fillPSetDescription(edm::ParameterSetDescription &desc);

  std::vector<std::vector<reco::TransientTrack> > clusterize(
      const std::vector<reco::TransientTrack> &tracks) const override;

  std::vector<TransientVertex> vertices(const std::vector<reco::TransientTrack> &tracks) const override;

  track_t fill(const std::vector<reco::TransientTrack> &tracks) const;

  void set_vtx_range(double beta, track_t &gtracks, vertex_t &gvertices) const;

  void clear_vtx_range(track_t &gtracks, vertex_t &gvertices) const;

  unsigned int thermalize(
      double beta, track_t &gtracks, vertex_t &gvertices, const double delta_max, const double rho0 = 0.) const;

  double update(
      double beta, track_t &gtracks, vertex_t &gvertices, const double rho0 = 0, const bool updateTc = false) const;

  void dump(const double beta, const vertex_t &y, const track_t &tks, const int verbosity = 0) const;
  bool zorder(vertex_t &y) const;
  bool find_nearest(double z, double t, vertex_t &y, unsigned int &k_min, double dz, double dt) const;
  bool merge(vertex_t &, track_t &, double &beta) const;
  bool purge(vertex_t &, track_t &, double &, const double) const;
  bool split(const double beta, track_t &t, vertex_t &y, double threshold = 1.) const;

  double beta0(const double betamax, track_t const &tks, vertex_t const &y) const;

  double get_Tc(const vertex_t &y, int k) const;
  void verify(const vertex_t &v, const track_t &tks, unsigned int nv = 999999, unsigned int nt = 999999) const;

private:
  double zdumpcenter_;
  double zdumpwidth_;

  double vertexSize_;
  double vertexSizeTime_;
  unsigned int maxIterations_;
  double coolingFactor_;
  double betamax_;
  double betastop_;
  double dzCutOff_;
  double d0CutOff_;
  double dtCutOff_;
  double t0Max_;

  double mintrkweight_;
  double uniquetrkweight_;
  double uniquetrkminp_;
  double zmerge_;
  double tmerge_;
  double betapurge_;

  unsigned int convergence_mode_;
  double delta_highT_;
  double delta_lowT_;

  double sel_zrange_;
  const double zrange_min_ = 0.1;  // smallest z-range to be included in a tracks cluster list
};

//#ifndef DAClusterizerInZT_vect_h
#endif
