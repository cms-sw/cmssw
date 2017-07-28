#ifndef UTILITIES_GENERAL_GRAPH_PATH_H
#define UTILITIES_GENERAL_GRAPH_PATH_H

#include "Utilities/General/interface/Graph.h"
#include <iostream>
#include <map>
#include <set>
#include <vector>

namespace cms::util {

  template <class N, class E>
    class GraphPath
  {
  public:
    using segment_type = std::pair<N,N>;
    //FIXME: GraphPath: find memory optimized representation of type paths_type ...
    using paths_type = std::map< segment_type, std::set< segment_type > > ;
    using paths_set = std::set< std::vector<N> >;
    using chain_type = std::vector< std::pair<N,N> >;
    using result_type = std::vector<std::vector<segment_type> >;
    GraphPath(const cms::util::Graph<N,E> & g, const N & root);
    ~GraphPath() {}
    bool fromTo(const N & from, const N & to, std::vector< std::vector<N> > & result) const;
  
    void calcPaths(const cms::util::Graph<N,E>& g, const N & root);
    void findSegments(const N & n, std::set< segment_type >& result);
    void stream(std::ostream&);
  
    /**
       - false, if no (directed!) path between fromTo.first and fromTo.second, p = empty set
       - true, if (directed!) paths exist between fromTo.first an fromTo.second, p = set of pathnodes
    */			     
    bool paths2(const segment_type & ft, result_type& result) const;			     

    //private:
    void update(segment_type & s, result_type & r, int pos) const;  
    paths_type paths_;
  };

  template <class N, class M>
    std::ostream & operator<<(std::ostream & os, const std::pair<N,M> & p)
  {
    os << p.first << ":" << p.second;
  }

  template <class N, class E>
    bool GraphPath<N,E>::fromTo(const N & from, const N & to, std::vector< std::vector<N> > & result) const
    {
      result_type tres;
      bool rslt=false;
      if (paths2(segment_type(from,to),tres)) {
	for ( auto const& rit : tres ) {
	  N & target = (*rit)[0].second;
	  typename std::vector<segment_type>::reverse_iterator pit = rit->rbegin();
	  typename std::vector<segment_type>::reverse_iterator pend = rit->rend(); 
	  --pend;
	  std::vector<N> v(1,(*rit)[0].first); // <A,X> -> <A>
	  //std::cout << pit->first << '-';
	  ++pit;
	  for(; pit!=pend; ++pit) {
	    v.push_back(pit->second);
	    //std::cout << pit->second << '-';
	  }  	 
	  //std::cout << target << std::endl;
	  v.push_back(target);
	  result.push_back(v);	 
	}
	rslt=true;
      }  
      return rslt;
    }
 
  template <class N, class E>
    bool GraphPath<N,E>::paths2(const segment_type & ft, result_type& result) const
    {
      typename paths_type::const_iterator git = paths_.find(ft);
      if (git==paths_.end()) {
	result.clear();
	return false;
      }
  
      std::vector<segment_type> v;
      v.push_back(git->first);
      result.push_back(v); // starting point; the set will be enlarged & the std::vectors inside
      // get pushed_back as new path-segments appear ...
  
      // find a possible direct-connetion:
      //set<segment_type>::iterator direct_it = 
  
      bool goOn(true);
  
      while(goOn) {
	//FIXME: use size_type whenever possible ..
	int u = result.size();
	int i;
	int cntdwn=u;
	for (i=0; i<u; ++i) {
	  segment_type & upd_seg = result[i].back();
	  if (upd_seg.first!=upd_seg.second) // only update result if not <X,X> !!
	    update(upd_seg,result,i); // adds new paths ..
	  else
	    --cntdwn;
	}
	goOn = bool(cntdwn);
    
	//std::cout << "0.--: cntdwn=" << cntdwn << std::endl;
	/* PRINT THE RESULT
	   result_type::iterator rit = result.begin();
	   for(; rit!=result.end(); ++rit) {
	   std::vector<segment_type>::iterator pit = rit->begin();
	   for(; pit!=rit->end(); ++pit) {
	   std::cout << "[" << pit->first << "," << pit->second << "] ";
	   }
	   std::cout << std::endl;
	   }  
	   std::cout << "===========" << std::endl;
	*/
      }    
      return true;
    } 

  template <class N, class E>
    void GraphPath<N,E>::update(segment_type & s, result_type & result, int u) const
    {
      // s ...      segment, which is used to find its children
      // result ... std::vector of path-std::vectors
      // u ...      path in result which is currently under observation, s is it's 'back'
      const std::set<segment_type> & segs = paths_.find(s)->second;
      typename std::set<segment_type>::const_iterator segit = segs.begin();

      if (segs.size()==0)  {
	std::cerr << "you should never get here: GraphPath::update(...)" << std::endl;
	exit(1);
      }  
      /*
	std::cout << "1. s=" << s.first << " " << s.second 
        << " aseg=" << segit->first << " " << segit->second << std::endl;
      */
      std::vector<segment_type>  temp_pth = result[u];
      ++segit;
      for (; segit!=segs.end(); ++segit) { // create new pathes (whenever a the path-tree is branching)
	std::vector<segment_type> v = temp_pth;
	v.push_back(*segit);
	result.push_back(v);     
      }
      temp_pth.push_back(*segs.begin()); // just append the first new segment to the existing one (also, when no branch!)
      result[u]=temp_pth;
    }


  template <class N, class E>
    GraphPath<N,E>::GraphPath(const cms::util::Graph<N,E>& g, const N & root)
    {
      calcPaths(g,root);
    }

  /** 
      creates a lookup-table of starting-points of pathes between nodes n A and B
      A->B: A->X, A->Y, B->B  (B->B denotes a direct connectino between A and B,
      A->X means that B can be reached from X directly while X can be reached from A)
      the lookup-table is stored in a std::map< pair<n,n>, set< pair<n,n> > (could be a multistd::map..)			   
  */  
  template <class N, class E>
    void GraphPath<N,E>::calcPaths(const cms::util::Graph<N,E>& g, const N & n)
    {
      // find n(ode) in g(raph) and all its children (get rid of the
      // multiconnections ...
      //set< pair<N,E> > nodes;
      std::pair<bool,typename cms::util::Graph<N,E>::neighbour_range> childRange = g.nodes(n);
      if (!childRange.first) 
	return;
   
      std::set<N> children;
      typename std::set< std::pair<N,E> >::const_iterator nit = childRange.second.first;
      for(; nit!=childRange.second.second; ++nit)
	children.insert(nit->first);
   
      // iterate over children and ..
      typename std::set<N>::iterator cit = children.begin();
      for(; cit!=children.end(); ++cit) {
	segment_type key = segment_type(n,*cit); // create new direct path-segment
	segment_type direct = segment_type(*cit,*cit);
	std::set< segment_type > temp;
	temp.insert(direct); // add direct connection as first member of set,
	// but not as <A,B> but as <B,B> to mark a direct connection
	//if(n != *cit) {			  
	paths_.insert(std::make_pair(key,temp));
	findSegments(n,temp); // look for previous segments leading to n
	typename std::set< segment_type >::iterator sit = temp.begin();
	for (; sit!=temp.end(); ++sit) { // iterator over already linked segments
	  if (sit->first != key.second) // don't insert <B,B> as key!
	    paths_[segment_type(sit->first,key.second)].insert(*sit);
	}
	//} 
	//calcPath(g,*cit);        
      }
      for(cit=children.begin();cit!=children.end();++cit) {
	//if (n != * cit)
	calcPaths(g,*cit);  
      }  
    }

  template <class N, class E>
    void GraphPath<N,E>::findSegments(const N & n, std::set< segment_type >& result)
    {
      typename paths_type::iterator pit = paths_.begin();
      for (; pit!=paths_.end(); ++pit) {
	if (pit->first.second == n)
	  result.insert(pit->first);
      }    
    }

  template <class N, class E>
    void GraphPath<N,E>::stream(std::ostream & os)
    {
      typename paths_type::iterator it = paths_.begin();
      for(; it!=paths_.end(); ++it) {
	os << "[" << it->first.first << "->" << it->first.second << "] : ";
	typename std::set<segment_type>::iterator sit = it->second.begin();
	os << "< ";
	for(; sit!=it->second.end();++sit) {
	  os << " [" << sit->first << "->" << sit->second << "] ";
	}  
	os << " >" << std::endl;  
    
      }
    }
 
} // namespace cms::util

#endif
