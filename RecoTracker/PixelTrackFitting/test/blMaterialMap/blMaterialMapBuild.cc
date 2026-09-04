// Turn the Geant4 step trees written by blMaterialMapRays_cfg.py (MaterialBudgetAction, AllStepsToTree)
// into the BL-fit material lattice rho(r,z) [X0/cm] on the 0.5 cm radial x 1 cm z grid of
// RecoTracker/PixelTrackFitting/interface/BLMaterialMap.h.
//
// For every step the tree gives its end points and dmb = dl/X0. A Geant4 step never crosses a volume
// boundary, so rho is constant along the step and the step is split exactly into the cells it crosses:
// the breakpoints are the roots of |P(t)|_T = r_b (quadratic) and z(t) = z_b (linear) for t in [0,1].
//
// Rays leave the origin flat in eta and phi, so with r = s sin(theta), z = s cos(theta) the Jacobian
// |d(r,z)/d(eta,s)| = r and the ray family deposits path measure per unit (r,z) area proportional to
// 1/r. Weighting every deposit by the (Simpson) mean radius of its sub-interval turns dr dz / r back
// into dr dz and makes the estimator the area average of the local 1/X0 over the cell:
//        num[cell] += (dmb/L) * dl * rbar          den[cell] += dl * rbar
//        rho[cell]  = num/den                       [X0^-1 per cm]
//
// Also accumulated, per 0.01-wide |eta| bin: the box-clipped sum(dmb) inside r < 125 cm, |z| < 280 cm.
//
// usage:  blMaterialMapBuild out.bin tree1.root [tree2.root ...]
// output: int kNR, int kNZ, kNR*kNZ doubles num, kNR*kNZ doubles den, one double = rays processed,
//         int kNE, kNE doubles etaSum, kNE doubles etaN.
// The per-job outputs are summed by blMaterialMapEmit.py, which writes the compiled-in table.
#include <TFile.h>
#include <TTree.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static const int kNR = 250;
static const int kNZ = 560;
static const double kDR = 0.5;
static const double kDZ = 1.0;
static const double kZMAX = 280.0;
static const int MAXSTEPS = 10000;

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s out.bin tree1.root [tree2.root ...]\n", argv[0]);
    return 1;
  }
  std::vector<double> num(kNR * kNZ, 0.0), den(kNR * kNZ, 0.0);
  // box-clipped sum(dmb) per |eta| bin
  static const int kNE = 600;  // |eta| 0 .. 6 in 0.01
  std::vector<double> etaSum(kNE, 0.0), etaN(kNE, 0.0);
  double nray = 0;

  static double iX[MAXSTEPS], iY[MAXSTEPS], iZ[MAXSTEPS];
  static double fX[MAXSTEPS], fY[MAXSTEPS], fZ[MAXSTEPS];
  static float dmb[MAXSTEPS];
  int nst = 0;

  std::vector<double> tb;  // breakpoints, reused
  tb.reserve(4096);

  for (int ifile = 2; ifile < argc; ++ifile) {
    TFile* f = TFile::Open(argv[ifile]);
    if (!f || f->IsZombie()) {
      fprintf(stderr, "cannot open %s\n", argv[ifile]);
      continue;
    }
    TTree* t = (TTree*)f->Get("T1");
    if (!t) {
      fprintf(stderr, "no T1 in %s\n", argv[ifile]);
      f->Close();
      continue;
    }
    t->SetBranchStatus("*", 0);
    const char* on[] = {"Nsteps", "DeltaMB", "Initial X", "Initial Y", "Initial Z", "Final X", "Final Y", "Final Z"};
    for (auto b : on)
      t->SetBranchStatus(b, 1);
    float pEta = 0.f;
    t->SetBranchStatus("Particle Eta", 1);
    t->SetBranchAddress("Particle Eta", &pEta);
    t->SetBranchAddress("Nsteps", &nst);
    t->SetBranchAddress("DeltaMB", dmb);
    t->SetBranchAddress("Initial X", iX);
    t->SetBranchAddress("Initial Y", iY);
    t->SetBranchAddress("Initial Z", iZ);
    t->SetBranchAddress("Final X", fX);
    t->SetBranchAddress("Final Y", fY);
    t->SetBranchAddress("Final Z", fZ);

    const Long64_t nent = t->GetEntries();
    for (Long64_t ie = 0; ie < nent; ++ie) {
      t->GetEntry(ie);
      nray += 1;
      double mbBox = 0.0;
      for (int is = 0; is < nst && is < MAXSTEPS; ++is) {
        // mm -> cm
        const double x0 = iX[is] * 0.1, y0 = iY[is] * 0.1, z0 = iZ[is] * 0.1;
        const double x1 = fX[is] * 0.1, y1 = fY[is] * 0.1, z1 = fZ[is] * 0.1;
        const double dx = x1 - x0, dy = y1 - y0, dz = z1 - z0;
        const double len = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (!(len > 0.0))
          continue;
        const double rho_step = dmb[is] / len;  // [X0^-1 cm^-1], constant on the step

        // exact cell breakpoints in t in (0,1)
        tb.clear();
        tb.push_back(0.0);
        tb.push_back(1.0);
        // z boundaries (linear)
        if (dz != 0.0) {
          double za = std::min(z0, z1), zb = std::max(z0, z1);
          int ka = (int)std::floor(za) + 1, kb = (int)std::ceil(zb) - 1;
          if (ka < -(int)kZMAX)
            ka = -(int)kZMAX;
          if (kb > (int)kZMAX)
            kb = (int)kZMAX;
          for (int k = ka; k <= kb; ++k) {
            double tt = (k - z0) / dz;
            if (tt > 0.0 && tt < 1.0)
              tb.push_back(tt);
          }
        }
        // r boundaries (quadratic: a t^2 + b t + c = 0 with c = r0^2 - rb^2)
        {
          const double a = dx * dx + dy * dy;
          const double b = 2.0 * (x0 * dx + y0 * dy);
          const double r0s = x0 * x0 + y0 * y0;
          const double r1s = x1 * x1 + y1 * y1;
          double rmin = std::sqrt(std::min(r0s, r1s)), rmax = std::sqrt(std::max(r0s, r1s));
          // the segment dips below both endpoints' radius when the parabola vertex is inside
          if (a > 0.0) {
            const double tv = -b / (2.0 * a);
            if (tv > 0.0 && tv < 1.0) {
              const double rv2 = r0s + b * tv + a * tv * tv;
              if (rv2 > 0 && std::sqrt(rv2) < rmin)
                rmin = std::sqrt(rv2);
            }
          }
          int ka = (int)std::floor(rmin / kDR) + 1, kb = (int)std::ceil(rmax / kDR) - 1;
          if (ka < 1)
            ka = 1;
          if (kb > kNR)
            kb = kNR;
          for (int k = ka; k <= kb; ++k) {
            const double rb = (double)k * kDR;
            const double c = r0s - rb * rb;
            if (a > 0.0) {
              const double disc = b * b - 4.0 * a * c;
              if (disc >= 0.0) {
                const double sq = std::sqrt(disc);
                double t1 = (-b - sq) / (2.0 * a), t2 = (-b + sq) / (2.0 * a);
                if (t1 > 0.0 && t1 < 1.0)
                  tb.push_back(t1);
                if (t2 > 0.0 && t2 < 1.0)
                  tb.push_back(t2);
              }
            } else if (b != 0.0) {
              const double tt = -c / b;
              if (tt > 0.0 && tt < 1.0)
                tb.push_back(tt);
            }
          }
        }
        std::sort(tb.begin(), tb.end());

        for (size_t j = 0; j + 1 < tb.size(); ++j) {
          const double ta = tb[j], tc = tb[j + 1];
          const double dt = tc - ta;
          if (dt <= 0.0)
            continue;
          const double tm = 0.5 * (ta + tc);
          const double xm = x0 + tm * dx, ym = y0 + tm * dy, zm = z0 + tm * dz;
          const double rm = std::sqrt(xm * xm + ym * ym);
          const int ir = (int)(rm / kDR);
          const int iz = (int)((zm + kZMAX) / kDZ);
          if (ir < 0 || ir >= kNR || iz < 0 || iz >= kNZ)
            continue;
          const double dl = dt * len;
          // Simpson mean radius over the sub-interval: the exact dr dz measure.
          const double xa = x0 + ta * dx, ya = y0 + ta * dy;
          const double xc = x0 + tc * dx, yc = y0 + tc * dy;
          const double rbar = (std::sqrt(xa * xa + ya * ya) + 4.0 * rm + std::sqrt(xc * xc + yc * yc)) / 6.0;
          const double w = dl * rbar;
          const size_t k = (size_t)ir * kNZ + iz;
          den[k] += w;
          num[k] += rho_step * w;
          mbBox += rho_step * dl;
        }
      }
      const int ieta = int(std::fabs((double)pEta) / 0.01);
      if (ieta >= 0 && ieta < kNE) {
        etaSum[ieta] += mbBox;
        etaN[ieta] += 1.0;
      }
    }
    f->Close();
    delete f;
    fprintf(stderr, "  %s : %lld rays cumulative %.0f\n", argv[ifile], (long long)nent, nray);
  }

  FILE* o = fopen(argv[1], "wb");
  int a = kNR, b = kNZ;
  fwrite(&a, sizeof(int), 1, o);
  fwrite(&b, sizeof(int), 1, o);
  fwrite(num.data(), sizeof(double), num.size(), o);
  fwrite(den.data(), sizeof(double), den.size(), o);
  fwrite(&nray, sizeof(double), 1, o);
  int ne = kNE;
  fwrite(&ne, sizeof(int), 1, o);
  fwrite(etaSum.data(), sizeof(double), etaSum.size(), o);
  fwrite(etaN.data(), sizeof(double), etaN.size(), o);
  fclose(o);
  fprintf(stderr, "wrote %s (%.0f rays)\n", argv[1], nray);
  return 0;
}
