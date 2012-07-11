#ifndef EtaPhiContainer_h
#define EtaPhiContainer_h

#include <vector>
#include <map>
#include <stdint.h>

#include "DataFormats/Common/interface/RefTraits.h"




template < typename T > class EtaPhiContainer
{
	friend std::ostream& operator<< ( std::ostream& aStream , const  EtaPhiContainer<T>& aContainer ){
		int i(0);
		for( typename EtaPhiContainer<T>::const_iterator lIt=aContainer.begin() ; lIt!=aContainer.end() ; ++lIt ){
			aStream << "Element " << (i++) << " :\n" << *lIt << "\n";
		}
		return aStream;
	}


  public:
	typedef typename std::vector < T >::iterator iterator;
	typedef typename std::vector < T >::const_iterator const_iterator;

	typedef T value_type;

  public:
	EtaPhiContainer(  ):
	mHash( std::vector < uint16_t > ( 16384, uint16_t( -1 ) ) )
	{
		mContainer.reserve( 16384 );
	}


	EtaPhiContainer( const EtaPhiContainer < T > &aEtaPhiContainer ):
	mContainer( aEtaPhiContainer.mContainer ),
	mHash( aEtaPhiContainer.mHash )
	{
		mContainer.reserve( 16384 );
	}


	virtual ~ EtaPhiContainer(  )
	{
	}

	T & at( const std::size_t & aIndex )
	{
		return mContainer.at( aIndex );
	}

	const T & at( const std::size_t & aIndex ) const
	{
		return mContainer.at( aIndex );
	}


	iterator find( const int &aEta, const int &aPhi )
	{
		uint16_t lIndex = mHash.at( hash( aEta, aPhi ) );

		if ( lIndex == uint16_t( -1 ) ){
			return mContainer.end(  );
		}
		return mContainer.begin(  ) + lIndex;
	}


	const_iterator find( const int &aEta, const int &aPhi )const
	{
		uint16_t lIndex = mHash.at( hash( aEta, aPhi ) );

		if ( lIndex == uint16_t( -1 ) ){
			return mContainer.end(  );
		}
		return mContainer.begin(  ) + lIndex;
	}


/*	
	std::vector< T >& operator->(){ return mContainer; }
	std::vector< T >& operator*(){ return mContainer; }
*/

	iterator begin(  )
	{
		return mContainer.begin(  );
	}

	const_iterator begin(  ) const
	{
		return mContainer.begin(  );
	}

	iterator end(  )
	{
		return mContainer.end(  );
	}

	const_iterator end(  ) const
	{
		return mContainer.end(  );
	}

	iterator insert( const int &aEta, const int &aPhi, const T & aT )
	{
		mHash.at( hash( aEta, aPhi ) ) = mContainer.size(  );
		mContainer.push_back( aT );
		return ( --mContainer.end(  ) );
	}

	std::size_t size(  )const
	{
		return mContainer.size(  );
	}


	void sort(  ){
		std::multimap < T , uint16_t > lMap;
		for( uint16_t lHash = 0 ; lHash != mHash.size(  ) ; ++lHash ){
			uint16_t lIndex = mHash.at( lHash );
			if( lIndex != uint16_t( -1 ) ){
				lMap.insert ( std::make_pair( *(mContainer.begin(  ) + lIndex) , lHash ) );
				mHash.at( lHash ) = uint16_t( -1 );
			}
		}

		mContainer.clear();

		for( typename std::multimap < T , uint16_t >::reverse_iterator lItr = lMap.rbegin(); lItr != lMap.rend(); ++lItr ){
			mHash.at( lItr->second ) = mContainer.size();
			mContainer.push_back( lItr->first );
		}
	}


  private:
	inline uint16_t hash( const int &aEta, const int &aPhi )const
	{
		return uint16_t( ( ( aEta & 0x7f ) << 7 ) | ( aPhi & 0x7f ) );
	}


  private:
	std::vector < T > mContainer;
	std::vector < uint16_t > mHash;
};




template < typename T > class EtaPhiContainerLookUp:public std::binary_function < const EtaPhiContainer < T > &, int, const T *>
{
  public:
	const T *operator(  ) ( const EtaPhiContainer < T > &aContainer, int aIndex )
	{
		return &aContainer.at( aIndex );
	}
};


namespace edm
{
	namespace refhelper
	{
		template < typename T > class FindTrait < EtaPhiContainer < T >, T >
		{
		  public:
			typedef EtaPhiContainerLookUp < T > value;
		};
	}
}

#endif
