#ifndef UTILITIES_GENERAL_GRAPH_UTIL_H
#define UTILITIES_GENERAL_GRAPH_UTIL_H

#include "Utilities/General/interface/Graph.h"
#include "Utilities/General/interface/GraphWalker.h"
#include <iostream>
#include <string>

template<class N, class E>
  void output(const cms::util::Graph<N,E> & g, const N & root)
{
  cms::util::GraphWalker<N,E> w(g,root);
  bool go=true;
  while(go) {
    std::cout << w.current().first << ' ';
    go=w.next();
  }
  std::cout << std::endl;
}

template<class N, class E>
  void graph_combine(const cms::util::Graph<N,E> & g1, const cms::util::Graph<N,E> & g2,
		     const N & n1, const N & n2, const N & root,
		     cms::util::Graph<N,E> & result)
{
  result = g1;
  result.replace(n1,n2);
  //output(result,n2);
  cms::util::GraphWalker<N,E> walker(g2,n2);
  while (walker.next()) {
    const N & parent = g2.nodeData((++walker.stack().rbegin())->first->first);
    /*
      N parent = g2.nodeData((++walker.stack().rbegin())->first->first);
      N child  = walker.current().first;
      E edge   = walker.current().second;
    */
    //std::cout << parent << ' ' << walker.current().first << ' ' << walker.current().second<< std::endl;
    result.addEdge(parent, walker.current().first, walker.current().second);
    //result.addEdge(parent,child,edge);
    //result.dump_graph();
    //output(result,n2);
  }
  result.replace(n2,root);  			
  //output(result,root);
}		

template<class N, class E>
  void graph_tree_output(const cms::util::Graph<N,E> & g, const N & root, std::ostream & os)
{
  cms::util::GraphWalker<N,E> w(g,root);
  bool go=true;
  unsigned int depth=0;
  while (go) {
    std::string s(2*depth,' ');
    os << ' ' << s << w.current().first << '(' << w.current().second << ')' << std::endl;
    go = w.firstChild();
    if( go ) {
      ++depth;
    }
    else if(w.stack().size() >1 && w.nextSibling()) {
      go=true;
    }
    else {
      go=false;
      while(w.parent()) {
	--depth;
	if (w.stack().size()>1 && w.nextSibling()) {
	  go=true;
	  break;
	}
      }
    }
  }
}

#endif
