#ifndef BaseBinFinder_H
#define BaseBinFinder_H

/** \class BaseBinFinder
 * Abstract interface for a bin finder.
 *
 *  $Date: 2005/09/21 10:16:30 $
 *  $Revision: 1.1 $
 *  \author N. Amapane - INFN Torino
 */

template <class T>
class BaseBinFinder {
public:
  
  BaseBinFinder() {};

  virtual ~BaseBinFinder(){}

  /// Return the index of bin at given position.
  virtual int binIndex( T pos) const =0;
  
  /// Returns an index in the valid range
  virtual int binIndex( int i) const =0;
  
  /// The middle of the ind-th bin
  virtual T binPosition( int ind) const = 0;

private:
  //  int theNbins;

};
#endif

