#ifndef NbSharedTracks_h
#define NbSharedTracks_h



namespace reco {

  //
  // forward declarations
  //
  class Vertex;


  //
  // this class
  //
  class NbSharedTracks {

  public:
    NbSharedTracks() {};
    ~NbSharedTracks() {};

    int sharedTracks(const reco::Vertex &v1,
		     const reco::Vertex &v2) const;
    
  private:
    
  }; // class
} // namespace

#endif
