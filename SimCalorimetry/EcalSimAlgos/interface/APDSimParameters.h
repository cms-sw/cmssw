#ifndef EcalSimAlgos_APDSimParameters_h
#define EcalSimAlgos_APDSimParameters_h

#include <cmath>
#include <string>
#include <vector>

class APDSimParameters {
public:
  APDSimParameters(bool addToBarrel,
                   bool separateDigi,
                   double simToPELow,
                   double simToPEHigh,
                   double timeOffset,
                   double timeOffWidth,
                   bool doPEStats,
                   const std::string& digiTag,
                   const std::vector<double>& nonlParms)
      :

        m_addToBarrel(addToBarrel),
        m_separateDigi(separateDigi),
        m_simToPELow(simToPELow),
        m_simToPEHigh(simToPEHigh),
        m_timeOffset(timeOffset),
        m_timeOffWidth(fabs(timeOffWidth)),
        m_doPEStats(doPEStats),
        m_digiTag(digiTag),
        m_nonlParms(nonlParms) {}

  virtual ~APDSimParameters() {}

  bool addToBarrel() const { return m_addToBarrel; }
  bool separateDigi() const { return m_separateDigi; }
  double simToPELow() const { return m_simToPELow; }
  double simToPEHigh() const { return m_simToPEHigh; }
  double timeOffset() const { return m_timeOffset; }
  double timeOffWidth() const { return m_timeOffWidth; }
  bool doPEStats() const { return m_doPEStats; }

  const std::string& digiTag() const { return m_digiTag; }

  const std::vector<double>& nonlParms() const { return m_nonlParms; }

private:
  bool m_addToBarrel;
  bool m_separateDigi;
  double m_simToPELow;
  double m_simToPEHigh;
  double m_timeOffset;
  double m_timeOffWidth;
  bool m_doPEStats;
  std::string m_digiTag;
  std::vector<double> m_nonlParms;
};

#endif
