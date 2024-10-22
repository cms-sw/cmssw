#ifndef MU_END_ANALOG_SIGNAL
#define MU_END_ANALOG_SIGNAL

/** \class CSCAnalogSignal
 *  Simple histogram meant to represent the analog
 *  signal on a detector element.
 *
 * \author Rick Wilkinson
 *
 * Last mod: <BR>
 * 30-Jun-00 ptc Add further traps in getBinValue() and setBinValue(). <BR>
 * 06-Jul-00 ptc In fact the getBinValue trap was an important bug-fix:
 * it trapped on > size() of stl std::vector but should have trapped >= size().
 * It occasionally does reach size(). <BR>
 * <p>
 * Mods (performace improvements) by Vin  31/07/2000<br>
 *   Critical methods (getBinValue, get Value +=) inlined<br>
 *   bin-size stored and used as his inverse
 *   (encapulation helped in not changing interface, named changed to use
 * compiler to catch its occurrencies)<br> swap input std::vector (be careful if
 * const..)<br> do proper interpolation (not just /2)<br>
 *
 */

#include <cassert>
#include <iosfwd>
#include <vector>

// TODO remove
#include <iostream>

class CSCAnalogSignal {
public:
  inline CSCAnalogSignal() : theElement(0), invBinSize(0.), theBinValues(0), theTotal(0), theTimeOffset(0.) {}

  inline CSCAnalogSignal(
      int element, float binSize, std::vector<float> &binValues, float total = 0., float timeOffset = 0.)
      : theElement(element), invBinSize(1. / binSize), theBinValues(), theTotal(total), theTimeOffset(timeOffset) {
    theBinValues.swap(binValues);
  }

  /// constructor from time and amp shape
  //  CSCAnalogSignal(int element, const CSCAnalogSignal& shape, float time,
  //  float total);

  inline int getElement() const { return theElement; };
  inline void setElement(int element) { theElement = element; };
  inline float getBinValue(int i) const {
    return (i >= static_cast<int>(theBinValues.size()) || i < 0) ? 0. : theBinValues[i];
  }

  inline float getValue(float t) const {
    // interpolate between bins, if necessary
    float retval = 0.;
    float f = (t - theTimeOffset) * invBinSize + 0.000000001;
    if (f >= 0.) {
      int i = static_cast<int>(f);
      f -= static_cast<float>(i);
      retval = (1. - f) * getBinValue(i) + f * getBinValue(i + 1);
    }
    return retval;
  }

  //  inline void  setBinValue(int i, float value) {
  //    if( i >= 0 && i < theBinValues.size() )
  //      theBinValues[i] = value;
  //  }

  inline int getSize() const { return theBinValues.size(); };
  inline float getBinSize() const { return 1. / invBinSize; };
  inline float getTotal() const { return theTotal; };
  inline float getTimeOffset() const { return theTimeOffset; };
  inline void setTimeOffset(float offset) { theTimeOffset = offset; };

  inline void superimpose(const CSCAnalogSignal &signal2) {
    size_t n = theBinValues.size();
    for (size_t i = 0; i < n; ++i) {
      float t = i / invBinSize + theTimeOffset;
      theBinValues[i] += signal2.getValue(t);
    }
    theTotal += signal2.theTotal;
  }

  inline void operator+=(float offset) {
    for (int i = 0; i < getSize(); ++i) {
      theBinValues[i] += offset;
    }
  }

  inline void operator*=(float scaleFactor) {
    for (int i = 0; i < getSize(); ++i) {
      theBinValues[i] *= scaleFactor;
    }
    theTotal *= scaleFactor;
  }

  friend std::ostream &operator<<(std::ostream &, const CSCAnalogSignal &);

  float &operator[](int i) {
    assert(i >= 0 && i < getSize());
    return theBinValues[i];
  }

  const float &operator[](int i) const {
    assert(i >= 0 && i < getSize());
    return theBinValues[i];
  }

  /// the time when the signal peaks
  float peakTime() const;
  unsigned size() const { return theBinValues.size(); }

private:
  int theElement;
  float invBinSize;
  std::vector<float> theBinValues;
  float theTotal;
  float theTimeOffset;
};

#endif
