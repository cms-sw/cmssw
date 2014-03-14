#ifndef XHistogram_h
#define XHistogram_h

#include <algorithm>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <boost/shared_ptr.hpp>

#include <TH2F.h>
#include <TH2I.h>

class XHistogram 
{
public:
  typedef TH2I                      ColorMap;
  typedef TH2F                      Histogram;
  typedef std::pair<double, double> Range;

protected:
  double m_minDl;
  Range  m_xRange;
  Range  m_yRange;
  size_t m_xBins;
  size_t m_yBins;
  size_t m_size;

  std::vector< boost::shared_ptr<Histogram> > m_histograms;
  boost::shared_ptr<Histogram> m_normalization;
  boost::shared_ptr<ColorMap>  m_colormap;
  boost::shared_ptr<Histogram> m_dummy;

public:
  /// default CTOR 
  XHistogram(void) :
    m_minDl( 0.000001 ),
    m_xRange(),
    m_yRange(),
    m_xBins(),
    m_yBins(),
    m_size(),
    m_histograms(),
    m_normalization(),
    m_colormap(),
    m_dummy()
  { }
 
  // explicit CTOR
  XHistogram( size_t size, size_t bins_x, size_t bins_y, Range x, Range y, size_t zones, std::vector<double> max ) :
    m_minDl( 0.000001 ),
    m_xRange( x ),
    m_yRange( y ),
    m_xBins( bins_x ),
    m_yBins( bins_y ),
    m_size( size ),
    m_histograms( m_size ),
    m_normalization(),
    m_colormap()
  {
    // setup unnamed ROOT histograms
    for (size_t i = 0; i < m_size; ++i) {
      m_histograms[i].reset(new Histogram( 0, 0, bins_x, x.first, x.second, bins_y, y.first, y.second ));
      m_histograms[i]->SetMinimum( 0. );
      m_histograms[i]->SetMaximum( max[i] );
    }
    m_normalization.reset(new Histogram( 0, 0, bins_x, x.first, x.second, bins_y, y.first, y.second ));
    m_colormap.reset(new ColorMap( 0, 0, bins_x, x.first, x.second, bins_y, y.first, y.second ));
    m_colormap->SetMinimum( 0 );
    m_colormap->SetMaximum( zones );
    Histogram( 0, 0, 0, 0., 0., 0, 0., 0. );        // make ROOT "forget" about unnamed histograms
  }

  void reset( void )
  {
    // reset all ROOT histograms
    for (size_t i = 0; i < m_size; ++i)
      m_histograms[i]->Reset("ICES");
    m_normalization->Reset();
    m_colormap->Reset("ICES");
  }

  /// fill one point
  void fill( double x, double y, const std::vector<double> & weight, double norm );
 
  /// fill one point and set its color
  void fill( double x, double y, const std::vector<double> & weight, double norm, unsigned int colour );
    
  /// fill one segment, normalizing each bin's weight to the fraction of the segment it contains
  void fill( const Range& x, const Range& y, const std::vector<double> & weight, double norm );
    
  /// fill one segment and set its color, normalizing each bin's weight to the fraction of the segment it contains
  void fill( const Range& x, const Range& y, const std::vector<double> & weight, double norm, unsigned int colour );

  /// normalize the histograms
  void normalize(void);

  /// access one of the histograms
  Histogram * get(size_t h = 0) const
  {
    if (h < m_size)
      return (Histogram *) m_histograms[h]->Clone(0);
    else
      return 0;
  }

  /// access the normalization
  Histogram * normalization(void) const
  {
    return (Histogram *) m_normalization->Clone(0);
  }

  /// access the colormap
  ColorMap * colormap(void) const
  {
    return (ColorMap *) m_colormap->Clone(0);
  }

  /// set the minimum length of sub-segment a segment should be split into:
  /// when splitting across bin boundaries with splitSegment(...), sub-segments shorter than this are skipped
  void setMinDl( double dl )
  {
    m_minDl = dl;
  }
 
protected:
  struct position {
    double f;
    double x;
    double y;

    position() : f(0), x(0), y(0) { }

    position(double f_, double x_, double y_) : f(f_), x(x_), y(y_) { }

    bool operator<(const position& other) const {
      return f < other.f;
    }
  };

  /// split a segment into a vector of points
  std::vector<position> splitSegment( Range x, Range y ) const;

  /// check the weights passed as an std::vector have the correct size
  void check_weight(const std::vector<double> & weight) throw (std::invalid_argument)
  {
    // run time check for vector size
    if (weight.size() != m_size)
       throw std::invalid_argument("weight: wrong number of elements");
  }

};

#endif // XHistogram_h
