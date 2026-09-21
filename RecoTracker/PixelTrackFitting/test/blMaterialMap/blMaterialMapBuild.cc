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
// The ionisation (dE/dx) weights ride on the same sub-intervals: the step's material is identified by its
// (density, X0) pair -- the only material identity the step tree carries -- in the Geant4 material table
// BLMaterialTableDump writes for the run, which gives rhoE = rho Z/A [mol/cm^3] (the Landau xi per cm of
// path), the mean excitation energy I and the plasma energy. Per cell:
//        numE  += rhoE * w        numEI += rhoE * ln(I/eV) * w        numEE += rhoE * ln(rhoE) * w
// so rhoE[cell] = numE/den is the area average of rho Z/A and numEI/numE, numEE/numE the rhoE-weighted
// (i.e. xi-weighted) log-means of I and of rho Z/A over the cell, the two quantities the Landau MPV of a
// composite column needs (the ln I term and, through the plasma energy, the density-effect term).
//
// Also accumulated, per 0.01-wide |eta| bin: the box-clipped sum(dmb) inside r < 125 cm, |z| < 280 cm.
//
// usage:  blMaterialMapBuild out.bin materials.txt tree1.root [tree2.root ...]
// output: int kNR, int kNZ, kNR*kNZ doubles num, kNR*kNZ doubles den,
//         3 x kNR*kNZ doubles numE, numEI, numEE, one double = rays processed,
//         int kNE, kNE doubles etaSum, kNE doubles etaN.
// The per-job outputs are summed by blMaterialMapEmit.py, which writes the binary map.
#include <TFile.h>
#include <TTree.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const int kNR = 250;
static const int kNZ = 560;
static const double kDR = 0.5;
static const double kDZ = 1.0;
static const double kZMAX = 280.0;
static const int MAXSTEPS = 10000;

// One Geant4 material of the run (BLMaterialTableDump line): identity by (density, X0), dE/dx weights.
struct Material {
  double density, x0, rhoE, lnI, lnRhoE;
};

// Look the step's material up by its (density, X0) pair: exact to the float the tree stores, else the
// nearest pair in relative distance (a Geant4 version can move a mixture's X0 at its last digit); steps
// whose nearest match is farther than kMatTol are counted as unmatched and still use it.
static const double kMatTol = 1e-3;

int main(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s out.bin materials.txt tree1.root [tree2.root ...]\n", argv[0]);
    return 1;
  }
  std::vector<Material> mats;
  {
    FILE* m = fopen(argv[2], "r");
    if (!m) {
      fprintf(stderr, "cannot open the material table %s\n", argv[2]);
      return 1;
    }
    char line[1024];
    while (fgets(line, sizeof(line), m)) {
      if (line[0] == '#')
        continue;
      unsigned idx;
      char name[512];
      double density, x0, rhoE, I;
      if (sscanf(line, "%u %511s %lf %lf %lf %lf", &idx, name, &density, &x0, &rhoE, &I) == 6 && rhoE > 0. && I > 0.)
        mats.push_back({density, x0, rhoE, std::log(I), std::log(rhoE)});
    }
    fclose(m);
    if (mats.empty()) {
      fprintf(stderr, "no materials in %s\n", argv[2]);
      return 1;
    }
  }
  std::vector<double> num(kNR * kNZ, 0.0), den(kNR * kNZ, 0.0);
  std::vector<double> numE(kNR * kNZ, 0.0), numEI(kNR * kNZ, 0.0), numEE(kNR * kNZ, 0.0);
  double nUnmatched = 0., nStepsAll = 0.;
  int lastMat = -1;  // steps of one ray mostly repeat the material: try the previous match first
  auto lookup = [&](float density, float x0) {
    auto rel = [&](const Material& mm) {
      return std::fabs(mm.density - density) / (std::fabs(mm.density) + 1e-30) +
             std::fabs(mm.x0 - x0) / (std::fabs(mm.x0) + 1e-30);
    };
    if (lastMat >= 0 && rel(mats[lastMat]) < 1e-6)
      return lastMat;
    int best = 0;
    double dbest = rel(mats[0]);
    for (size_t i = 1; i < mats.size(); ++i) {
      const double d = rel(mats[i]);
      if (d < dbest) {
        dbest = d;
        best = int(i);
      }
    }
    if (dbest > kMatTol)
      nUnmatched += 1.;
    lastMat = best;
    return best;
  };
  // box-clipped sum(dmb) per |eta| bin
  static const int kNE = 600;  // |eta| 0 .. 6 in 0.01
  std::vector<double> etaSum(kNE, 0.0), etaN(kNE, 0.0);
  double nray = 0;

  static double iX[MAXSTEPS], iY[MAXSTEPS], iZ[MAXSTEPS];
  static double fX[MAXSTEPS], fY[MAXSTEPS], fZ[MAXSTEPS];
  static float dmb[MAXSTEPS];
  static float mDens[MAXSTEPS], mX0[MAXSTEPS];
  int nst = 0;

  std::vector<double> tb;  // breakpoints, reused
  tb.reserve(4096);

  for (int ifile = 3; ifile < argc; ++ifile) {
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
    const char* on[] = {"Nsteps",
                        "DeltaMB",
                        "Initial X",
                        "Initial Y",
                        "Initial Z",
                        "Final X",
                        "Final Y",
                        "Final Z",
                        "Material Density",
                        "Material X0"};
    for (auto b : on)
      t->SetBranchStatus(b, 1);
    t->SetBranchAddress("Material Density", mDens);
    t->SetBranchAddress("Material X0", mX0);
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
        // the tree's material X0 is in Geant4 mm (density already in g/cm^3)
        const Material& mat = mats[lookup(mDens[is], 0.1f * mX0[is])];
        nStepsAll += 1.;

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
          numE[k] += mat.rhoE * w;
          numEI[k] += mat.rhoE * mat.lnI * w;
          numEE[k] += mat.rhoE * mat.lnRhoE * w;
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
  fwrite(numE.data(), sizeof(double), numE.size(), o);
  fwrite(numEI.data(), sizeof(double), numEI.size(), o);
  fwrite(numEE.data(), sizeof(double), numEE.size(), o);
  fwrite(&nray, sizeof(double), 1, o);
  int ne = kNE;
  fwrite(&ne, sizeof(int), 1, o);
  fwrite(etaSum.data(), sizeof(double), etaSum.size(), o);
  fwrite(etaN.data(), sizeof(double), etaN.size(), o);
  fclose(o);
  fprintf(stderr,
          "wrote %s (%.0f rays, %zu materials, %.0f of %.0f steps beyond the (density, X0) match tolerance)\n",
          argv[1],
          nray,
          mats.size(),
          nUnmatched,
          nStepsAll);
  return 0;
}
