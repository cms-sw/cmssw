#ifndef CaloSimAlgos_NoiseStatistics_h
#define CaloSimAlgos_NoiseStatistics_h

/** For validation purposes.  This program
    calculates mean and RMS of a distribution

  \Author Rick Wilkinson
*/
#include <string>
#include <iosfwd>

class NoiseStatistics
{
public:
  NoiseStatistics(std::string name, double expectedMean, double expectedRMS);
  /// prints to LogInfo upon destruction
  ~NoiseStatistics();

  void addEntry(double value, double weight=1.);

  std::string name() const {return name_;}

  double mean() const;

  double RMS() const;

  double weightedMean() const;

  double expectedMean() const {return expectedMean_;}

  double expectedRMS() const {return expectedRMS_;}

  unsigned long nEntries() const {return n_;}

private:

  std::string name_;
  double expectedMean_;
  double expectedRMS_;
  double sum_;
  double sumOfSquares_;
  double weightedSum_;
  double sumOfWeights_;
  unsigned long n_;
};

std::ostream & operator<<(std::ostream & os, const NoiseStatistics & stat);

#endif

