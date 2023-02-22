#include "SimCalorimetry/EcalSimAlgos/interface/EcalShapeBase.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <cmath>
#include <algorithm>

#include <iostream>

EcalShapeBase::~EcalShapeBase() {}

EcalShapeBase::EcalShapeBase(bool useDBShape)
    : m_useDBShape(useDBShape),
      m_firstIndexOverThreshold(0),
      m_firstTimeOverThreshold(0.0),
      m_indexOfMax(0),
      m_timeOfMax(0.0),
      m_thresh(0.0) {}

void EcalShapeBase::setEventSetup(const edm::EventSetup& evtSetup, bool normalize) { buildMe(&evtSetup, normalize); }

double EcalShapeBase::timeOfThr() const { return m_firstTimeOverThreshold; }

double EcalShapeBase::timeOfMax() const { return m_timeOfMax; }

double EcalShapeBase::timeToRise() const { return timeOfMax() - timeOfThr(); }

double EcalShapeBase::threshold() const { return m_thresh; }

void EcalShapeBase::buildMe(const edm::EventSetup* evtSetup, bool normalize) {
  DVec shapeArray;

  float time_interval = 0;
  fillShape(time_interval,
            m_thresh,
            shapeArray,
            evtSetup);              // pure virtual function, implementation may vary for EB/EE/APD ...
  m_arraySize = shapeArray.size();  // original data

  m_denseArraySize = 10 * m_arraySize;  // dense array with interpolation between data
  m_kNBinsPerNSec =
      (unsigned int)(10 /
                     time_interval);  // used to be an unsigned int = 10 in  < CMSSW10X, should work for time intervals ~0.1, 0.2, 0.5, 1
  m_qNSecPerBin = time_interval / 10.;

  m_deriv.resize(m_denseArraySize);
  m_shape.resize(m_denseArraySize);

  const double maxel(*max_element(shapeArray.begin(), shapeArray.end()));

  const double maxelt(1.e-5 < maxel ? maxel : 1);

  if (normalize) {
    for (unsigned int i(0); i != shapeArray.size(); ++i) {
      shapeArray[i] = shapeArray[i] / maxelt;
    }
  }

  const double thresh(threshold() / maxelt);

  const double delta(m_qNSecPerBin / 2.);

  for (unsigned int denseIndex(0); denseIndex != m_denseArraySize; ++denseIndex) {
    const double xb((denseIndex + 0.5) * m_qNSecPerBin);

    const unsigned int ibin(denseIndex / 10);

    double value = 0.0;
    double deriv = 0.0;

    if (0 == ibin || shapeArray.size() == 1 + ibin)  // cannot do quadratic interpolation at ends
    {
      value = shapeArray[ibin];
      deriv = 0.0;
    } else {
      const double x(xb - (ibin + 0.5) * time_interval);
      const double f1(shapeArray[ibin - 1]);
      const double f2(shapeArray[ibin]);
      const double f3(shapeArray[ibin + 1]);
      const double a(f2);
      const double b((f3 - f1) / (2. * time_interval));
      const double c(((f1 + f3) / 2. - f2) / (time_interval * time_interval));
      value = a + b * x + c * x * x;
      deriv = (b + 2 * c * x) / delta;
    }

    m_shape[denseIndex] = value;
    m_deriv[denseIndex] = deriv;

    if (0 < denseIndex && thresh < value && 0 == m_firstIndexOverThreshold) {
      m_firstIndexOverThreshold = denseIndex - 1;
      m_firstTimeOverThreshold = m_firstIndexOverThreshold * m_qNSecPerBin;
    }

    if (m_shape[m_indexOfMax] < value) {
      m_indexOfMax = denseIndex;
    }
  }
  m_timeOfMax = m_indexOfMax * m_qNSecPerBin;
}

unsigned int EcalShapeBase::timeIndex(double aTime) const {
  const int index(m_firstIndexOverThreshold + (unsigned int)(aTime * m_kNBinsPerNSec + 0.5));

  const bool bad((int)m_firstIndexOverThreshold > index || (int)m_denseArraySize <= index);

  if ((int)m_denseArraySize <= index) {
    LogDebug("EcalShapeBase") << " ECAL MGPA shape requested for out of range time " << aTime;
  }
  return (bad ? m_denseArraySize : (unsigned int)index);
}

double EcalShapeBase::operator()(double aTime) const {
  // return pulse amplitude for request time in ns

  const unsigned int index(timeIndex(aTime));
  return (m_denseArraySize == index ? 0 : m_shape[index]);
}

double EcalShapeBase::derivative(double aTime) const {
  const unsigned int index(timeIndex(aTime));
  return (m_denseArraySize == index ? 0 : m_deriv[index]);
}

void EcalShapeBase::m_shape_print(const char* fileName) const {
  std::ofstream fs;
  fs.open(fileName);
  fs << "{\n";
  for (auto i : m_shape)
    fs << "vec.push_back(" << i << ");\n";
  fs << "}\n";
  fs.close();
}
