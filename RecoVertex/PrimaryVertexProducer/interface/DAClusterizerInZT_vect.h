#ifndef DAClusterizerInZT_vect_h
#define DAClusterizerInZT_vect_h

/**\class DAClusterizerInZT_vect

 Description: separates event tracks into clusters along the beam line

	Version which auto-vectorizes with gcc 4.6 or newer

 */

#include "RecoVertex/PrimaryVertexProducer/interface/TrackClusterizerInZ.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <vector>
#include "DataFormats/Math/interface/Error.h"
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"

#include <memory>

class DAClusterizerInZT_vect final : public TrackClusterizerInZ {
public:
  // Internal data structure to
  struct track_t {
    void addItem(
        double new_z, double new_t, double new_dz2, double new_dt2, const reco::TransientTrack *new_tt, double new_pi) {
      z.push_back(new_z);
      t.push_back(new_t);
      dz2.push_back(new_dz2);
      dt2.push_back(new_dt2);
      tt.push_back(new_tt);

      pi.push_back(new_pi);  // track weight
      Z_sum.push_back(1.0);  // Z[i]   for DA clustering, initial value as done in ::fill
      kmin.push_back(0);
      kmax.push_back(0);
    }

    unsigned int getSize() const { return z.size(); }

    // has to be called everytime the items are modified
    void extractRaw() {
      z_ptr = &z.front();
      t_ptr = &t.front();
      dz2_ptr = &dz2.front();
      dt2_ptr = &dt2.front();
      pi_ptr = &pi.front();
      Z_sum_ptr = &Z_sum.front();
    }

    double *z_ptr;   // z-coordinate at point of closest approach to the beamline
    double *t_ptr;   // t-coordinate at point of closest approach to the beamline
    double *pi_ptr;  // track weight

    double *dz2_ptr;    // square of the error of z(pca)
    double *dt2_ptr;    // square of the error of t(pca)
    double *Z_sum_ptr;  // Z[i]   for DA clustering

    std::vector<double> z;      // z-coordinate at point of closest approach to the beamline
    std::vector<double> t;      // t-coordinate at point of closest approach to the beamline
    std::vector<double> dz2;    // square of the error of z(pca)
    std::vector<double> dt2;    // square of the error of t(pca)
    std::vector<double> Z_sum;  // Z[i]   for DA clustering
    std::vector<double> pi;     // track weight
    std::vector<unsigned int> kmin;
    std::vector<unsigned int> kmax;
    std::vector<const reco::TransientTrack *> tt;  // a pointer to the Transient Track
  };

  struct vertex_t {
    void addItem(double new_z, double new_t, double new_pk) {
      z.push_back(new_z);
      t.push_back(new_t);
      pk.push_back(new_pk);

      ei_cache.push_back(0.0);
      ei.push_back(0.0);
      swz.push_back(0.0);
      swt.push_back(0.0);
      se.push_back(0.0);
      nuz.push_back(0.0);
      nut.push_back(0.0);
      szz.push_back(0.0);
      stt.push_back(0.0);
      szt.push_back(0.0);

      dt2.push_back(0.0);
      sumw.push_back(0.0);

      extractRaw();
    }

    unsigned int getSize() const { return z.size(); }

    // has to be called everytime the items are modified
    void extractRaw() {
      z_ptr = &z.front();
      t_ptr = &t.front();
      pk_ptr = &pk.front();
      dt2_ptr = &dt2.front();
      sumw_ptr = &sumw.front();

      ei_ptr = &ei.front();
      swz_ptr = &swz.front();
      swt_ptr = &swt.front();
      se_ptr = &se.front();
      nuz_ptr = &nuz.front();
      nut_ptr = &nut.front();
      szz_ptr = &szz.front();
      stt_ptr = &stt.front();
      szt_ptr = &szt.front();

      ei_cache_ptr = &ei_cache.front();
    }

    void insertItem(unsigned int k, double new_z, double new_t, double new_pk, track_t &tks) {
      z.insert(z.begin() + k, new_z);
      t.insert(t.begin() + k, new_t);
      pk.insert(pk.begin() + k, new_pk);
      dt2.insert(dt2.begin() + k, 0.0);
      sumw.insert(sumw.begin() + k, 0.0);

      ei_cache.insert(ei_cache.begin() + k, 0.0);
      ei.insert(ei.begin() + k, 0.0);
      swz.insert(swz.begin() + k, 0.0);
      swt.insert(swt.begin() + k, 0.0);
      se.insert(se.begin() + k, 0.0);

      nuz.insert(nuz.begin() + k, 0.0);
      nut.insert(nut.begin() + k, 0.0);
      szz.insert(szz.begin() + k, 0.0);
      stt.insert(stt.begin() + k, 0.0);
      szt.insert(szt.begin() + k, 0.0);

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
      z.erase(z.begin() + k);
      t.erase(t.begin() + k);
      pk.erase(pk.begin() + k);
      dt2.erase(dt2.begin() + k);
      sumw.erase(sumw.begin() + k);

      ei_cache.erase(ei_cache.begin() + k);
      ei.erase(ei.begin() + k);
      swz.erase(swz.begin() + k);
      swt.erase(swt.begin() + k);
      se.erase(se.begin() + k);

      nuz.erase(nuz.begin() + k);
      nut.erase(nut.begin() + k);
      szz.erase(szz.begin() + k);
      stt.erase(stt.begin() + k);
      szt.erase(szt.begin() + k);

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

    unsigned int insertOrdered(double z, double t, double pk, track_t &tks) {
      // insert a new cluster according to it's z-position, return the index at which it was inserted

      unsigned int k = 0;
      for (; k < getSize(); k++) {
        if (z < z_ptr[k])
          break;
      }
      insertItem(k, z, t, pk, tks);
      return k;
    }

    void debugOut() {
      std::cout << "vertex_t size: " << getSize() << std::endl;

      for (unsigned int i = 0; i < getSize(); ++i) {
        std::cout << " z = " << z_ptr[i] << " t = " << t_ptr[i] << " pk = " << pk_ptr[i] << std::endl;
      }
    }

    std::vector<double> z;     // z coordinate
    std::vector<double> t;     // t coordinate
    std::vector<double> pk;    // vertex weight for "constrained" clustering
    std::vector<double> dt2;   // only used with vertex time uncertainties
    std::vector<double> sumw;  // only used with vertex time uncertainties

    double *z_ptr;
    double *t_ptr;
    double *pk_ptr;
    double *dt2_ptr;
    double *sumw_ptr;

    double *ei_cache_ptr;
    double *ei_ptr;
    double *swz_ptr;
    double *swt_ptr;
    double *se_ptr;
    double *szz_ptr;
    double *stt_ptr;
    double *szt_ptr;
    double *nuz_ptr;
    double *nut_ptr;

    // --- temporary numbers, used during update
    std::vector<double> ei_cache;
    std::vector<double> ei;
    std::vector<double> swz;
    std::vector<double> swt;
    std::vector<double> se;
    std::vector<double> nuz;
    std::vector<double> nut;
    std::vector<double> szz;
    std::vector<double> stt;
    std::vector<double> szt;

    // copy made at the beginning of thermalize
    std::vector<double> z0;  //           z coordinate at last vtx range fixing
  };

  DAClusterizerInZT_vect(const edm::ParameterSet &conf);

  std::vector<std::vector<reco::TransientTrack> > clusterize(
      const std::vector<reco::TransientTrack> &tracks) const override;

  std::vector<TransientVertex> vertices(const std::vector<reco::TransientTrack> &tracks, const int verbosity = 0) const;

  track_t fill(const std::vector<reco::TransientTrack> &tracks) const;

  void set_vtx_range(double beta, track_t &gtracks, vertex_t &gvertices) const;

  void clear_vtx_range(track_t &gtracks, vertex_t &gvertices) const;

  unsigned int thermalize(
      double beta, track_t &gtracks, vertex_t &gvertices, const double delta_max, const double rho0 = 0.) const;

  double update(double beta, track_t &gtracks, vertex_t &gvertices, const double rho0 = 0) const;

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
  bool verbose_;
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
