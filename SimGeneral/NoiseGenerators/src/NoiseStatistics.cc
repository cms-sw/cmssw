#include "SimGeneral/NoiseGenerators/interface/NoiseStatistics.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <cmath>
#include <iostream>

NoiseStatistics::NoiseStatistics(std::string name, double expectedMean, double expectedRMS) 
:  name_(name),
   expectedMean_(expectedMean),
   expectedRMS_(expectedRMS),
   sum_(0.),
   sumOfSquares_(0.),
   weightedSum_(0.),
   sumOfWeights_(0.),
   n_(0)
{
}


NoiseStatistics::~NoiseStatistics() 
{
  std::cout << *this << std::endl;
}

void NoiseStatistics::addEntry(double value, double weight) {
  sum_ += value;
  sumOfSquares_ += (value*value);
  weightedSum_ += value*weight;
  sumOfWeights_ += weight;
  ++n_;
}


double NoiseStatistics::mean() const {
  return sum_/n_;
}


double NoiseStatistics::RMS() const {
  double numerator = sumOfSquares_ - sum_*sum_/n_;
  unsigned long denominator = n_-1;
  return std::sqrt(numerator/denominator);
}

 
double NoiseStatistics::weightedMean() const  {
  return weightedSum_ / sumOfWeights_;
}


std::ostream& operator<<(std::ostream & os,const NoiseStatistics & stat) {
  os << "OVAL " << stat.name() << " entries:" << stat.nEntries();
  if(stat.nEntries() > 0) {
     os << " Mean: " << stat.mean() 
        << " (expect " << stat.expectedMean() << ")";
  }
  if(stat.nEntries() > 1) {      
		 os << "  RMS: " << stat.RMS()
        << " (expect " << stat.expectedRMS() << ")";
  }
  return os;
}


