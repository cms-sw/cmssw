///////////////////////////////////////////////////////////////////////////////
// File: HcalQie.cc
// Description: Simulation of QIE readout for HCal
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/HcalQie.h"

#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Random/RandPoissonQ.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"

#include <iostream>
#include <iomanip>
#include <cmath>

//#define EDM_ML_DEBUG

HcalQie::HcalQie(edm::ParameterSet const& p) {
  //static SimpleConfigurable<double> p1(4.0,   "HcalQie:qToPE");
  //static SimpleConfigurable<int>    p2(6,     "HcalQie:BinOfMax");
  //static SimpleConfigurable<int>    p3(2,     "HcalQie:SignalBuckets");
  //static SimpleConfigurable<int>    p4(0,     "HcalQie:PreSamples");
  //static SimpleConfigurable<int>    p5(10,    "HcalQie:NumOfBuckets");
  //static SimpleConfigurable<double> p6(0.5,   "HcalQie:SigmaNoise");
  //static SimpleConfigurable<double> p7(0.0005,"HcalQie:EDepPerPE");
  //static SimpleConfigurable<int>    p8(4,     "HcalQie:BaseLine");

  edm::ParameterSet m_HQ = p.getParameter<edm::ParameterSet>("HcalQie");
  qToPE = m_HQ.getParameter<double>("qToPE");
  binOfMax = m_HQ.getParameter<int>("BinOfMax");
  signalBuckets = m_HQ.getParameter<int>("SignalBuckets");
  preSamples = m_HQ.getParameter<int>("PreSamples");
  numOfBuckets = m_HQ.getParameter<int>("NumOfBuckets");
  sigma = m_HQ.getParameter<double>("SigmaNoise");
  eDepPerPE = m_HQ.getParameter<double>("EDepPerPE");
  int bl = m_HQ.getParameter<int>("BaseLine");

  shape_ = shape();
  code_ = code();
  charge_ = charge();
  if (signalBuckets == 1) {
    phase_ = -3;
    rescale_ = 1.46;
  } else if (signalBuckets == 3) {
    phase_ = -1;
    rescale_ = 1.06;
  } else if (signalBuckets == 4) {
    phase_ = 0;
    rescale_ = 1.03;
  } else {
    phase_ = -2;
    rescale_ = 1.14;
    signalBuckets = 2;
  }
  weight_ = weight(binOfMax, signalBuckets, preSamples, numOfBuckets);
  baseline = codeToQ(bl);
  bmin_ = binOfMax - 3;
  if (bmin_ < 0)
    bmin_ = 0;
  if (binOfMax > numOfBuckets)
    bmax_ = numOfBuckets + 5;
  else
    bmax_ = binOfMax + 5;

  edm::LogVerbatim("HcalSim") << "HcalQie: initialized with binOfMax " << binOfMax << " sample from " << bmin_ << " to "
                              << bmax_ << "; signalBuckets " << signalBuckets << " Baseline/Phase/Scale " << baseline
                              << "/" << phase_ << "/" << rescale_ << "\n                          Noise " << sigma
                              << "fC  fCToPE " << qToPE << " EDepPerPE " << eDepPerPE;
}

HcalQie::~HcalQie() { edm::LogVerbatim("HcalSim") << "HcalQie:: Deleting Qie"; }

std::vector<double> HcalQie::shape() {
  // pulse shape time constants in ns
  const float ts1 = 8.;  // scintillation time constants : 1,2,3
  const float ts2 = 10.;
  const float ts3 = 29.3;
  const float thpd = 4.;  // HPD current collection drift time
  const float tpre = 5.;  // preamp time constant

  const float wd1 = 2.;  // relative weights of decay exponents
  const float wd2 = 0.7;
  const float wd3 = 1.;

  // HPD starts at I and rises to 2I in thpd of time
  double norm = 0.0;
  int j, hpd_siz = (int)(thpd);
  std::vector<double> hpd_drift(hpd_siz);
  for (j = 0; j < hpd_siz; j++) {
    double tmp = (double)j + 0.5;
    hpd_drift[j] = 1.0 + tmp / thpd;
    norm += hpd_drift[j];
  }
  // normalize integrated current to 1.0
  for (j = 0; j < hpd_siz; j++) {
    hpd_drift[j] /= norm;
  }

  // Binkley shape over 6 time constants
  int preamp_siz = (int)(6 * tpre);
  std::vector<double> preamp(preamp_siz);
  norm = 0;
  for (j = 0; j < preamp_siz; j++) {
    double tmp = (double)j + 0.5;
    preamp[j] = tmp * exp(-(tmp * tmp) / (tpre * tpre));
    norm += preamp[j];
  }
  // normalize pulse area to 1.0
  for (j = 0; j < preamp_siz; j++) {
    preamp[j] /= norm;
  }

  // ignore stochastic variation of photoelectron emission
  // <...>
  // effective tile plus wave-length shifter decay time over 4 time constants

  int tmax = 6 * (int)ts3;
  std::vector<double> scnt_decay(tmax);
  norm = 0;
  for (j = 0; j < tmax; j++) {
    double tmp = (double)j + 0.5;
    scnt_decay[j] = wd1 * exp(-tmp / ts1) + wd2 * exp(-tmp / ts2) + wd3 * exp(-tmp / ts3);
    norm += scnt_decay[j];
  }
  // normalize pulse area to 1.0
  for (j = 0; j < tmax; j++) {
    scnt_decay[j] /= norm;
  }

  int nsiz = tmax + hpd_siz + preamp_siz + 1;
  std::vector<double> pulse(nsiz, 0.0);  // zeroing output pulse shape
  norm = 0;
  int i, k;
  for (i = 0; i < tmax; i++) {
    int t1 = i;  // and ignore jitter from optical path length
    for (j = 0; j < hpd_siz; j++) {
      int t2 = t1 + j;
      for (k = 0; k < preamp_siz; k++) {
        int t3 = t2 + k;
        float tmp = scnt_decay[i] * hpd_drift[j] * preamp[k];
        pulse[t3] += tmp;
        norm += tmp;
      }
    }
  }

  // normalize for 1 GeV pulse height
  edm::LogVerbatim("HcalSim") << "HcalQie: Convoluted Shape ============== Normalisation " << norm;
  for (i = 0; i < nsiz; i++) {
    pulse[i] /= norm;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HcalSim") << "HcalQie: Pulse[" << std::setw(3) << i << "] " << std::setw(8) << pulse[i];
#endif
  }

  return pulse;
}

std::vector<int> HcalQie::code() {
  unsigned int CodeFADCdata[122] = {
      0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,
      21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,
      44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  66,
      67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,
      88,  89,  90,  91,  92,  93,  94,  95,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
      111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127};

  std::vector<int> temp(122);
  int i;
  for (i = 0; i < 122; i++)
    temp[i] = (int)CodeFADCdata[i];

#ifdef EDM_ML_DEBUG
  int siz = temp.size();
  edm::LogVerbatim("HcalSim") << "HcalQie: Codes in array of size " << siz;
  for (i = 0; i < siz; i++)
    edm::LogVerbatim("HcalSim") << "HcalQie: Code[" << std::setw(3) << i << "] " << std::setw(6) << temp[i];
#endif
  return temp;
}

std::vector<double> HcalQie::charge() {
  double ChargeFADCdata[122] = {
      -1.5,   -0.5,   0.5,    1.5,    2.5,    3.5,    4.5,    5.5,    6.5,    7.5,    8.5,    9.5,    10.5,   11.5,
      12.5,   13.5,   14.5,   16.5,   18.5,   20.5,   22.5,   24.5,   26.5,   28.5,   31.5,   34.5,   37.5,   40.5,
      44.5,   48.5,   52.5,   57.5,   62.5,   67.5,   72.5,   77.5,   82.5,   87.5,   92.5,   97.5,   102.5,  107.5,
      112.5,  117.5,  122.5,  127.5,  132.5,  142.5,  152.5,  162.5,  172.5,  182.5,  192.5,  202.5,  217.5,  232.5,
      247.5,  262.5,  282.5,  302.5,  322.5,  347.5,  372.5,  397.5,  422.5,  447.5,  472.5,  497.5,  522.5,  547.5,
      572.5,  597.5,  622.5,  647.5,  672.5,  697.5,  722.5,  772.5,  822.5,  872.5,  922.5,  972.5,  1022.5, 1072.5,
      1147.5, 1222.5, 1297.5, 1372.5, 1472.5, 1572.5, 1672.5, 1797.5, 1922.5, 2047.5, 2172.5, 2297.5, 2422.5, 2547.5,
      2672.5, 2797.5, 2922.5, 3047.5, 3172.5, 3397.5, 3422.5, 3547.5, 3672.5, 3922.5, 4172.5, 4422.5, 4672.5, 4922.5,
      5172.5, 5422.5, 5797.5, 6172.5, 6547.5, 6922.5, 7422.5, 7922.5, 8422.5, 9047.5};

  std::vector<double> temp(122);
  int i;
  for (i = 0; i < 122; i++)
    temp[i] = (double)(ChargeFADCdata[i]);

#ifdef EDM_ML_DEBUG
  int siz = temp.size();
  edm::LogVerbatim("HcalSim") << "HcalQie: Charges in array of size " << siz;
  for (i = 0; i < siz; i++)
    edm::LogVerbatim("HcalSim") << "HcalQie: Charge[" << std::setw(3) << i << "] " << std::setw(8) << temp[i];
#endif
  return temp;
}

std::vector<double> HcalQie::weight(int binOfMax, int mode, int npre, int bucket) {
  std::vector<double> temp(bucket, 0);
  int i;
  for (i = binOfMax - 1; i < binOfMax + mode - 1; i++)
    temp[i] = 1.;
  if (npre > 0) {
    for (i = 0; i < npre; i++) {
      int j = binOfMax - 2 - i;
      temp[j] = -(double)mode / (double)npre;
    }
  }

#ifdef EDM_ML_DEBUG
  int siz = temp.size();
  edm::LogVerbatim("HcalSim") << "HcalQie: Weights in array of size " << siz << " and Npre " << npre;
  for (i = 0; i < siz; i++)
    edm::LogVerbatim("HcalSim") << "HcalQie: [Weight[" << i << "] = " << temp[i];
#endif
  return temp;
}

double HcalQie::codeToQ(int ic) {
  double tmp = 0;
  for (unsigned int i = 0; i < code_.size(); i++) {
    if (ic == code_[i]) {
      double delta;
      if (i == code_.size() - 1)
        delta = charge_[i] - charge_[i - 1];
      else
        delta = charge_[i + 1] - charge_[i];
      tmp = charge_[i] + 0.5 * delta;
      break;
    }
  }

  return tmp;
}

int HcalQie::getCode(double charge) {
  int tmp = 0;
  for (unsigned int i = 0; i < charge_.size(); i++) {
    if (charge < charge_[i]) {
      if (i > 0)
        tmp = code_[i - 1];
      break;
    }
  }

  return tmp;
}

double HcalQie::getShape(double time) {
  double tmp = 0;
  int k = (int)(time + 0.5);
  if (k >= 0 && k < ((int)(shape_.size()) - 1))
    tmp = 0.5 * (shape_[k] + shape_[k + 1]);

  return tmp;
}

std::vector<int> HcalQie::getCode(int nht, const std::vector<CaloHit>& hitbuf, CLHEP::HepRandomEngine* engine) {
  const double bunchSpace = 25.;
  int nmax = (bmax_ > numOfBuckets ? bmax_ : numOfBuckets);
  std::vector<double> work(nmax);

  // Noise in the channel
  for (int i = 0; i < numOfBuckets; i++)
    work[i] = CLHEP::RandGaussQ::shoot(engine, baseline, sigma);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalSim") << "HcalQie::getCode: Noise with baseline " << baseline << " width " << sigma << " and "
                              << nht << " hits";
  for (int i = 0; i < numOfBuckets; i++)
    edm::LogVerbatim("HcalSim") << "HcalQie: Code[" << i << "] = " << work[i];
  double etot = 0, esum = 0, photons = 0;
#endif
  if (nht > 0) {
    // Sort the hits
    std::vector<const CaloHit*> hits(nht);
    std::vector<const CaloHit*>::iterator k1, k2;
    int kk;
    for (kk = 0; kk < nht; kk++) {
      hits[kk] = &hitbuf[kk];
    }
    sort(hits.begin(), hits.end(), CaloHitMore());

    // Energy deposits
    for (kk = 0, k1 = hits.begin(); k1 != hits.end(); kk++, k1++) {
      double ehit = (**k1).e();
      double jitter = (**k1).t();
      int jump = 0;
      for (k2 = k1 + 1; k2 != hits.end() && (jitter - (**k2).t()) < 1. && (jitter - (**k2).t()) > -1.; k2++) {
        ehit += (**k2).e();
        jump++;
      }

      double avpe = ehit / eDepPerPE;
      CLHEP::RandPoissonQ randPoissonQ(*engine, avpe);
      double photo = randPoissonQ.fire();
#ifdef EDM_ML_DEBUG
      etot += ehit;
      photons += photo;
      edm::LogVerbatim("HcalSim") << "HcalQie::getCode: Hit " << kk << ":" << kk + jump << " Energy deposit " << ehit
                                  << " Time " << jitter << " Average and true no of PE " << avpe << " " << photo;
#endif
      double bintime = jitter - phase_ - bunchSpace * (binOfMax - bmin_);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HcalSim") << "HcalQie::getCode: phase " << phase_ << " binTime " << bintime;
#endif
      std::vector<double> binsum(nmax, 0);
      double norm = 0, sum = 0.;
      for (int i = bmin_; i < bmax_; i++) {
        bintime += bunchSpace;
        for (int j = 0; j < (int)(bunchSpace); j++) {
          double tim = bintime + j;
          double tmp = getShape(tim);
          binsum[i] += tmp;
        }
        sum += binsum[i];
      }

      if (sum > 0)
        norm = (photo / (sum * qToPE));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HcalSim") << "HcalQie::getCode: PE " << photo << " Sum " << sum << " Norm. " << norm;
#endif
      for (int i = bmin_; i < bmax_; i++)
        work[i] += binsum[i] * norm;

      kk += jump;
      k1 += jump;
    }
  }

  std::vector<int> temp(numOfBuckets, 0);
  for (int i = 0; i < numOfBuckets; i++) {
    temp[i] = getCode(work[i]);
#ifdef EDM_ML_DEBUG
    esum += work[i];
#endif
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalSim") << "HcalQie::getCode: Input " << etot << " GeV; Photons " << photons << ";  Output "
                              << esum << " fc";
#endif
  return temp;
}

double HcalQie::getEnergy(const std::vector<int>& code) {
  std::vector<double> work(numOfBuckets);
  double sum = 0;
  for (int i = 0; i < numOfBuckets; i++) {
    work[i] = codeToQ(code[i]);
    sum += work[i] * weight_[i];
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HcalSim") << "HcalQie::getEnergy: " << i << " code " << code[i] << " PE " << work[i];
#endif
  }

  double tmp;
  if (preSamples == 0) {
    tmp = (sum - signalBuckets * baseline) * rescale_ * qToPE * eDepPerPE;
  } else {
    tmp = sum * rescale_ * qToPE;
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalSim") << "HcalQie::getEnergy: PE " << sum * qToPE << " Energy " << tmp << " GeV";
#endif
  return tmp;
}
