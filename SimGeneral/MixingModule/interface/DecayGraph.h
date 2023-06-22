#ifndef SimGeneral_MixingModule_DecayGraph_h
#define SimGeneral_MixingModule_DecayGraph_h

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimDataFormats/Track/interface/SimTrack.h"

#if DEBUG
// boost optional (used by boost graph) results in some false positives with
// // -Wmaybe-uninitialized
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

// BOOST GRAPH LIBRARY
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/graphviz.hpp>

using boost::add_edge;
using boost::adjacency_list;
using boost::directedS;
using boost::edge;
using boost::edge_weight;
using boost::edge_weight_t;
using boost::listS;
using boost::property;
using boost::vecS;
using boost::vertex;
using boost::vertex_name;
using boost::vertex_name_t;

/* GRAPH DEFINITIONS

   The graphs represent the full decay chain.

   The parent-child relationship is the natural one, following "time".

   Each edge has a property (edge_weight_t) that holds a const pointer to the
   SimTrack that connects the 2 vertices of the edge, the number of simHits
   associated to that simTrack and the cumulative number of simHits of itself
   and of all its children. Only simHits within the selected detectors are
   taken into account. The cumulative property is filled during the dfs
   exploration of the graph: if not explored the number is 0.

   Each vertex has a property (vertex_name_t) that holds a const pointer to the
   SimTrack that originated that vertex and the cumulative number of simHits of
   all its outgoing edges. The cumulative property is filled during the dfs
   exploration of the graph: if not explored the number is 0.

   Stable particles are recovered/added in a second iterations and are linked
   to ghost vertices with an offset starting from the highest generated vertex.

   Multiple decays of a single particle that retains its original trackId are
   merged into one unique vertex (the first encountered) in order to avoid
   multiple counting of its associated simHits (if any).

*/
struct EdgeProperty {
  EdgeProperty(const SimTrack *t, int h, int c) : simTrack(t), simHits(h), cumulative_simHits(c) {}
  const SimTrack *simTrack;
  int simHits;
  int cumulative_simHits;
};

struct VertexProperty {
  VertexProperty() : simTrack(nullptr), cumulative_simHits(0) {}
  VertexProperty(const SimTrack *t, int c) : simTrack(t), cumulative_simHits(c) {}
  VertexProperty(const VertexProperty &other)
      : simTrack(other.simTrack), cumulative_simHits(other.cumulative_simHits) {}
  const SimTrack *simTrack;
  int cumulative_simHits;
};

using EdgeParticleClustersProperty = property<edge_weight_t, EdgeProperty>;
using VertexMotherParticleProperty = property<vertex_name_t, VertexProperty>;
using DecayChain = adjacency_list<listS, vecS, directedS, VertexMotherParticleProperty, EdgeParticleClustersProperty>;

namespace {
  extern const std::string messageCategoryGraph_;

  template <typename Edge, typename Graph, typename Visitor>
  void accumulateSimHits_edge(Edge &e, const Graph &g, Visitor *v) {
    auto const edge_property = get(edge_weight, g, e);
    v->total_simHits += edge_property.simHits;
    IfLogDebug(DEBUG, messageCategoryGraph_)
        << " Examining edges " << e << " --> particle " << edge_property.simTrack->type() << "("
        << edge_property.simTrack->trackId() << ")"
        << " with SimClusters: " << edge_property.simHits << " Accumulated SimClusters: " << v->total_simHits
        << std::endl;
  }
  template <typename Vertex, typename Graph>
  void print_vertex(Vertex &u, const Graph &g) {
    auto const vertex_property = get(vertex_name, g, u);
    IfLogDebug(DEBUG, messageCategoryGraph_) << " At " << u;
    // The Mother of all vertices has **no** SimTrack associated.
    if (vertex_property.simTrack)
      IfLogDebug(DEBUG, messageCategoryGraph_) << " [" << vertex_property.simTrack->type() << "]"
                                               << "(" << vertex_property.simTrack->trackId() << ")";
    IfLogDebug(DEBUG, messageCategoryGraph_) << std::endl;
  }

// Graphviz output functions will only be generated in DEBUG mode
#if DEBUG
  std::string graphviz_vertex(const VertexProperty &v) {
    std::ostringstream oss;
    oss << "{id: " << (v.simTrack ? v.simTrack->trackId() : 0) << ",\\ntype: " << (v.simTrack ? v.simTrack->type() : 0)
        << ",\\nchits: " << v.cumulative_simHits << "}";
    return oss.str();
  }

  std::string graphviz_edge(const EdgeProperty &e) {
    std::ostringstream oss;
    oss << "[" << (e.simTrack ? e.simTrack->trackId() : 0) << "," << (e.simTrack ? e.simTrack->type() : 0) << ","
        << e.simHits << "," << e.cumulative_simHits << "]";
    return oss.str();
  }
#endif

  class SimHitsAccumulator_dfs_visitor : public boost::default_dfs_visitor {
  public:
    int total_simHits = 0;
    template <typename Edge, typename Graph>
    void examine_edge(Edge e, const Graph &g) {
      accumulateSimHits_edge(e, g, this);
    }
    template <typename Edge, typename Graph>
    void finish_edge(Edge e, const Graph &g) {
      auto const edge_property = get(edge_weight, g, e);
      auto src = source(e, g);
      auto trg = target(e, g);
      auto cumulative = edge_property.simHits + get(vertex_name, g, trg).cumulative_simHits +
                        (get(vertex_name, g, src).simTrack ? get(vertex_name, g, src).cumulative_simHits
                                                           : 0);  // when we hit the root vertex we have to stop
                                                                  // adding back its contribution.
      auto const src_vertex_property = get(vertex_name, g, src);
      put(get(vertex_name, const_cast<Graph &>(g)), src, VertexProperty(src_vertex_property.simTrack, cumulative));
      put(get(edge_weight, const_cast<Graph &>(g)),
          e,
          EdgeProperty(edge_property.simTrack, edge_property.simHits, cumulative));
      IfLogDebug(DEBUG, messageCategoryGraph_)
          << " Finished edge: " << e << " Track id: " << get(edge_weight, g, e).simTrack->trackId()
          << " has accumulated " << cumulative << " hits" << std::endl;
      IfLogDebug(DEBUG, messageCategoryGraph_) << " SrcVtx: " << src << "\t" << get(vertex_name, g, src).simTrack
                                               << "\t" << get(vertex_name, g, src).cumulative_simHits << std::endl;
      IfLogDebug(DEBUG, messageCategoryGraph_) << " TrgVtx: " << trg << "\t" << get(vertex_name, g, trg).simTrack
                                               << "\t" << get(vertex_name, g, trg).cumulative_simHits << std::endl;
    }
  };

  using Selector = std::function<bool(EdgeProperty &)>;
}  // namespace

#endif
