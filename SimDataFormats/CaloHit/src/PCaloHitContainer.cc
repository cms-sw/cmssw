
#include <SimDataFormats/CaloHit/interface/PCaloHitContainer.h>

using namespace edm;

void PCaloHitContainer::insertHits (PCaloHitSingleContainer & p)
{
  m_data = p ;
}


void PCaloHitContainer::insertHit (PCaloHit & p)
{
  m_data.push_back (p) ;
}


PCaloHit PCaloHitContainer::operator[] (int i) const  
{
  return m_data[i] ;
}


void PCaloHitContainer::clear ()
{
  m_data.clear () ;
}


unsigned int PCaloHitContainer::size () const 
{
  return m_data.size () ;
}


std::vector<PCaloHit>::const_iterator PCaloHitContainer::begin () const 
{
  return m_data.begin () ;
}


std::vector<PCaloHit>::const_iterator PCaloHitContainer::end () const 
{
  return m_data.end () ;
}
