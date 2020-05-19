#ifndef DAClusterizerInZ_vect_h
#define DAClusterizerInZ_vect_h

/**\class DAClusterizerInZ_vect

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

class DAClusterizerInZ_vect final : public TrackClusterizerInZ {
public:
  // Internal data structure to
  struct track_t {
    void addItem(double new_z, double new_dz2, const reco::TransientTrack *new_tt, double new_pi) {
      z.push_back(new_z);
      dz2.push_back(new_dz2);
      tt.push_back(new_tt);

      pi.push_back(new_pi);  // track weight
      Z_sum.push_back(1.0);  // Z[i]   for DA clustering, initial value as done in ::fill

      kmin.push_back(0);
      kmax.push_back(0);
    }

    void addItemSorted(double new_z, double new_dz2, const reco::TransientTrack *new_tt, double new_pi) {
      // sort tracks with decreasing resolution (note that dz2 = 1/sigma^2)
      unsigned int i = 0;
      for (i = 0; i < z.size(); i++) {
        if (new_dz2 > dz2[i])
          break;
      }
      insertItem(i, new_z, new_dz2, new_tt, new_pi);
    }

    void insertItem(unsigned int i, double new_z, double new_dz2, const reco::TransientTrack *new_tt, double new_pi) {
      z.insert(z.begin() + i, new_z);
      dz2.insert(dz2.begin() + i, new_dz2);
      tt.insert(tt.begin() + i, new_tt);
      pi.insert(pi.begin() + i, new_pi);  // track weight

      Z_sum.insert(Z_sum.begin() + i, 1.0);  // Z[i]   for DA clustering, initial value as done in ::fill
      kmin.insert(kmin.begin() + i, 0);
      kmax.insert(kmax.begin() + i, 0);
    }

    unsigned int getSize() const { return z.size(); }

    // has to be called everytime the items are modified
    void extractRaw() {
      z_ptr = &z.front();
      dz2_ptr = &dz2.front();
      Z_sum_ptr = &Z_sum.front();
      pi_ptr = &pi.front();
    }

    double *__restrict__ z_ptr;    // z-coordinate at point of closest approach to the beamline
    double *__restrict__ dz2_ptr;  // square of the error of z(pca)

    double *__restrict__ Z_sum_ptr;  // Z[i]   for DA clustering
    double *__restrict__ pi_ptr;     // track weight

    std::vector<double> z;      // z-coordinate at point of closest approach to the beamline
    std::vector<double> dz2;    // square of the error of z(pca)
    std::vector<double> Z_sum;  // Z[i]   for DA clustering
    std::vector<double> pi;     // track weight
    std::vector<unsigned int> kmin;
    std::vector<unsigned int> kmax;
    std::vector<const reco::TransientTrack *> tt;  // a pointer to the Transient Track
  };

  struct vertex_t {
    std::vector<double> z;   //           z coordinate
    std::vector<double> pk;  //           vertex weight for "constrained" clustering

    // --- temporary numbers, used during update
    std::vector<double> ei_cache;
    std::vector<double> ei;
    std::vector<double> sw;
    std::vector<double> swz;
    std::vector<double> se;
    std::vector<double> swE;

    unsigned int getSize() const { return z.size(); }

    void addItem(double new_z, double new_pk) {
      z.push_back(new_z);
      pk.push_back(new_pk);

      ei_cache.push_back(0.0);
      ei.push_back(0.0);
      sw.push_back(0.0);
      swz.push_back(0.0);
      se.push_back(0.0);
      swE.push_back(0.0);

      extractRaw();
    }

    void insertItem(unsigned int k, double new_z, double new_pk) {
      z.insert(z.begin() + k, new_z);
      pk.insert(pk.begin() + k, new_pk);

      ei_cache.insert(ei_cache.begin() + k, 0.0);
      ei.insert(ei.begin() + k, 0.0);
      sw.insert(sw.begin() + k, 0.0);
      swz.insert(swz.begin() + k, 0.0);
      se.insert(se.begin() + k, 0.0);
      swE.insert(swE.begin() + k, 0.0);

      extractRaw();
    }

    void insertItem(unsigned int k, double new_z, double new_pk, track_t &tks) {
      z.insert(z.begin() + k, new_z);
      pk.insert(pk.begin() + k, new_pk);

      ei_cache.insert(ei_cache.begin() + k, 0.0);
      ei.insert(ei.begin() + k, 0.0);
      sw.insert(sw.begin() + k, 0.0);
      swz.insert(swz.begin() + k, 0.0);
      se.insert(se.begin() + k, 0.0);
      swE.insert(swE.begin() + k, 0.0);

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
      pk.erase(pk.begin() + k);

      ei_cache.erase(ei_cache.begin() + k);
      ei.erase(ei.begin() + k);
      sw.erase(sw.begin() + k);
      swz.erase(swz.begin() + k);
      se.erase(se.begin() + k);
      swE.erase(swE.begin() + k);

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

    void DebugOut() {
      std::cout << "vertex_t size: " << getSize() << std::endl;

      for (unsigned int i = 0; i < getSize(); ++i) {
        std::cout << " z = " << z_ptr[i] << " pk = " << pk_ptr[i] << std::endl;
      }
    }

    // has to be called everytime the items are modified
    void extractRaw() {
      z_ptr = &z.front();
      pk_ptr = &pk.front();

      ei_ptr = &ei.front();
      sw_ptr = &sw.front();
      swz_ptr = &swz.front();
      se_ptr = &se.front();
      swE_ptr = &swE.front();
      ei_cache_ptr = &ei_cache.front();
    }

    double *__restrict__ z_ptr;
    double *__restrict__ pk_ptr;

    double *__restrict__ ei_cache_ptr;
    double *__restrict__ ei_ptr;
    double *__restrict__ sw_ptr;
    double *__restrict__ swz_ptr;
    double *__restrict__ se_ptr;
    double *__restrict__ swE_ptr;
  };

  DAClusterizerInZ_vect(const edm::ParameterSet &conf);

  std::vector<std::vector<reco::TransientTrack> > clusterize(
      const std::vector<reco::TransientTrack> &tracks) const override;

  std::vector<TransientVertex> vertices(const std::vector<reco::TransientTrack> &tracks, const int verbosity = 0) const;

  track_t fill(const std::vector<reco::TransientTrack> &tracks) const;

  void set_vtx_range(double beta, track_t &gtracks, vertex_t &gvertices) const;

  void clear_vtx_range(track_t &gtracks, vertex_t &gvertices) const;

  unsigned int thermalize(
      double beta, track_t &gtracks, vertex_t &gvertices, const double delta_max, const double rho0 = 0.) const;

  double update(double beta, track_t &gtracks, vertex_t &gvertices, const double rho0 = 0) const;
  double updateTc(double beta, track_t &gtracks, vertex_t &gvertices, const double rho0 = 0) const;

  void dump(const double beta, const vertex_t &y, const track_t &tks, const int verbosity = 0) const;
  bool merge(vertex_t &y, track_t &tks, double &beta) const;
  bool purge(vertex_t &, track_t &, double &, const double) const;
  bool split(const double beta, track_t &t, vertex_t &y, double threshold = 1.) const;

  double beta0(const double betamax, track_t const &tks, vertex_t const &y) const;
  double evalF(const double beta, track_t const &tks, vertex_t const &v) const;
  void verify(const vertex_t &v, const track_t &tks, unsigned int nv = 999999, unsigned int nt = 999999) const;

private:
  bool verbose_;
  double zdumpcenter_;
  double zdumpwidth_;

  double vertexSize_;
  unsigned int maxIterations_;
  double coolingFactor_;
  double betamax_;
  double betastop_;
  double dzCutOff_;
  double d0CutOff_;

  double mintrkweight_;
  double uniquetrkweight_;
  double zmerge_;
  double betapurge_;

  unsigned int convergence_mode_;
  double delta_highT_;
  double delta_lowT_;

  double sel_zrange_;
  const double zrange_min_ = 0.1;  // smallest z-range to be included in a tracks cluster list
};

//#ifndef DAClusterizerInZ_vect_h
#endif
