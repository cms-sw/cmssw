// Turn the Geant4 step trees of blMaterialMapRays_cfg.py into the sums of the material lattice rho(r,z) [X0/cm]
// of interface/BLMaterialMap.h. Split each step into the (r,z) cells it crosses, and weight each part by its mean
// radius. Then num/den is the area average of 1/X0 in the cell.
//
// usage:  blMaterialMapBuild out.bin materials.txt tree1.root [tree2.root ...]
// output: int kNR, int kNZ, kNR*kNZ doubles num, kNR*kNZ doubles den,
//         3 x kNR*kNZ doubles numE, numEI, numEE, one double = rays processed.
// blMaterialMapEmit.py adds the outputs of the jobs and writes the binary map.
#include <TFile.h>
#include <TTree.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMap.h"

static const int kNR = blMaterialMap::kNR;
static const int kNZ = blMaterialMap::kNZ;
static const double kDR = blMaterialMap::kDR;
static const double kDZ = blMaterialMap::kDZ;
static const double kZMAX = blMaterialMap::kZMAX;
static const int MAXSTEPS = 10000;

// One Geant4 material of the run (BLMaterialTableDump line): identity by (density, X0), dE/dx weights.
struct Material {
  double density, x0, rhoE, lnI, lnRhoE;
};

// A table line is the index, the name, which can contain spaces, and these many numbers, from density on.
static const size_t kMatNumbers = 11;

// Find the step's material by its (density, X0) pair: the exact pair, else the nearest one in relative distance.
// The job fails if the nearest pair is farther than kMatTol.
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
      if (line[0] == '#') {
        continue;
      }
      std::istringstream words(line);
      std::vector<std::string> fields;
      for (std::string word; words >> word;) {
        fields.push_back(word);
      }
      if (fields.size() < kMatNumbers + 2) {
        fprintf(stderr, "malformed line in %s: %s", argv[2], line);
        return 1;
      }
      auto number = [&](size_t i) { return std::atof(fields[fields.size() - kMatNumbers + i].c_str()); };
      const double density = number(0), x0 = number(1), rhoE = number(2), I = number(3);
      if (rhoE > 0. && I > 0.) {
        mats.push_back({density, x0, rhoE, std::log(I), std::log(rhoE)});
      }
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
    if (lastMat >= 0 && rel(mats[lastMat]) < 1e-6) {
      return lastMat;
    }
    int best = 0;
    double dbest = rel(mats[0]);
    for (size_t i = 1; i < mats.size(); ++i) {
      const double d = rel(mats[i]);
      if (d < dbest) {
        dbest = d;
        best = int(i);
      }
    }
    if (dbest > kMatTol) {
      nUnmatched += 1.;
    }
    lastMat = best;
    return best;
  };
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
      return 1;
    }
    TTree* t = (TTree*)f->Get("T1");
    if (!t) {
      fprintf(stderr, "no T1 in %s\n", argv[ifile]);
      return 1;
    }
    t->SetBranchStatus("*", false);
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
    for (auto b : on) {
      t->SetBranchStatus(b, true);
    }
    t->SetBranchAddress("Material Density", mDens);
    t->SetBranchAddress("Material X0", mX0);
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
      if (t->GetEntry(ie) <= 0) {
        fprintf(stderr, "cannot read entry %lld of %s\n", (long long)ie, argv[ifile]);
        return 1;
      }
      nray += 1;
      for (int is = 0; is < nst && is < MAXSTEPS; ++is) {
        // mm -> cm
        const double x0 = iX[is] * 0.1, y0 = iY[is] * 0.1, z0 = iZ[is] * 0.1;
        const double x1 = fX[is] * 0.1, y1 = fY[is] * 0.1, z1 = fZ[is] * 0.1;
        const double dx = x1 - x0, dy = y1 - y0, dz = z1 - z0;
        const double len = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (!(len > 0.0)) {
          continue;
        }
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
          int ka = (int)std::floor(za / kDZ) + 1, kb = (int)std::ceil(zb / kDZ) - 1;
          if (ka < -(int)(kZMAX / kDZ)) {
            ka = -(int)(kZMAX / kDZ);
          }
          if (kb > (int)(kZMAX / kDZ)) {
            kb = (int)(kZMAX / kDZ);
          }
          for (int k = ka; k <= kb; ++k) {
            double tt = (k * kDZ - z0) / dz;
            if (tt > 0.0 && tt < 1.0) {
              tb.push_back(tt);
            }
          }
        }
        // r boundaries (quadratic: a t^2 + b t + c = 0 with c = r0^2 - rb^2)
        {
          const double a = dx * dx + dy * dy;
          const double b = 2.0 * (x0 * dx + y0 * dy);
          const double r0s = x0 * x0 + y0 * y0;
          const double r1s = x1 * x1 + y1 * y1;
          double rmin = std::sqrt(std::min(r0s, r1s)), rmax = std::sqrt(std::max(r0s, r1s));
          // the minimum radius is inside the segment if the parabola vertex is
          if (a > 0.0) {
            const double tv = -b / (2.0 * a);
            if (tv > 0.0 && tv < 1.0) {
              const double rv2 = r0s + b * tv + a * tv * tv;
              if (rv2 > 0 && std::sqrt(rv2) < rmin) {
                rmin = std::sqrt(rv2);
              }
            }
          }
          int ka = (int)std::floor(rmin / kDR) + 1, kb = (int)std::ceil(rmax / kDR) - 1;
          if (ka < 1) {
            ka = 1;
          }
          if (kb > kNR) {
            kb = kNR;
          }
          for (int k = ka; k <= kb; ++k) {
            const double rb = (double)k * kDR;
            const double c = r0s - rb * rb;
            if (a > 0.0) {
              const double disc = b * b - 4.0 * a * c;
              if (disc >= 0.0) {
                const double sq = std::sqrt(disc);
                double t1 = (-b - sq) / (2.0 * a), t2 = (-b + sq) / (2.0 * a);
                if (t1 > 0.0 && t1 < 1.0) {
                  tb.push_back(t1);
                }
                if (t2 > 0.0 && t2 < 1.0) {
                  tb.push_back(t2);
                }
              }
            } else if (b != 0.0) {
              const double tt = -c / b;
              if (tt > 0.0 && tt < 1.0) {
                tb.push_back(tt);
              }
            }
          }
        }
        std::sort(tb.begin(), tb.end());

        for (size_t j = 0; j + 1 < tb.size(); ++j) {
          const double ta = tb[j], tc = tb[j + 1];
          const double dt = tc - ta;
          if (dt <= 0.0) {
            continue;
          }
          const double tm = 0.5 * (ta + tc);
          const double xm = x0 + tm * dx, ym = y0 + tm * dy, zm = z0 + tm * dz;
          const double rm = std::sqrt(xm * xm + ym * ym);
          const int ir = (int)(rm / kDR);
          const int iz = (int)((zm + kZMAX) / kDZ);
          if (ir < 0 || ir >= kNR || iz < 0 || iz >= kNZ) {
            continue;
          }
          const double dl = dt * len;
          // Simpson mean radius over the sub-interval
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
        }
      }
    }
    f->Close();
    delete f;
    fprintf(stderr, "  %s : %lld rays cumulative %.0f\n", argv[ifile], (long long)nent, nray);
  }

  if (nUnmatched > 0.) {
    fprintf(stderr, "%.0f of %.0f steps are not in %s\n", nUnmatched, nStepsAll, argv[2]);
    return 1;
  }
  FILE* o = fopen(argv[1], "wb");
  if (!o) {
    fprintf(stderr, "cannot write %s\n", argv[1]);
    return 1;
  }
  int a = kNR, b = kNZ;
  fwrite(&a, sizeof(int), 1, o);
  fwrite(&b, sizeof(int), 1, o);
  fwrite(num.data(), sizeof(double), num.size(), o);
  fwrite(den.data(), sizeof(double), den.size(), o);
  fwrite(numE.data(), sizeof(double), numE.size(), o);
  fwrite(numEI.data(), sizeof(double), numEI.size(), o);
  fwrite(numEE.data(), sizeof(double), numEE.size(), o);
  fwrite(&nray, sizeof(double), 1, o);
  fclose(o);
  fprintf(stderr, "wrote %s (%.0f rays, %zu materials)\n", argv[1], nray, mats.size());
  return 0;
}
