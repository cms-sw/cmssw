//
// class declaration
//

struct MyGEMDigi
{
   Int_t detId;
   Short_t region, ring, station, layer, chamber, roll;
   Short_t strip, bx;
   Float_t x, y;
   Float_t g_r, g_eta, g_phi, g_x, g_y, g_z;
};

struct MyGEMCSCPadDigis
{
  Int_t detId;
  Short_t region, ring, station, layer, chamber, roll;
  Short_t pad, bx;
  Float_t x, y;
  Float_t g_r, g_eta, g_phi, g_x, g_y, g_z;
};

struct MyGEMCSCCoPadDigis
{
  Int_t detId;
  Short_t region, ring, station, layer, chamber, roll;
  Short_t pad, bx;
  Float_t x, y;
  Float_t g_r, g_eta, g_phi, g_x, g_y, g_z;
};

struct MySimTrack
{
  Float_t pt, eta, phi;
  Char_t charge;
  Char_t endcap;
  Char_t gem_sh_layer1, gem_sh_layer2;
  Char_t gem_dg_layer1, gem_dg_layer2;
  Char_t gem_pad_layer1, gem_pad_layer2;
  Float_t gem_sh_eta, gem_sh_phi;
  Float_t gem_sh_x, gem_sh_y;
  Float_t gem_dg_eta, gem_dg_phi;
  Float_t gem_pad_eta, gem_pad_phi;
  Float_t gem_lx_even, gem_ly_even;
  Float_t gem_lx_odd, gem_ly_odd;
  Char_t  has_gem_sh_l1, has_gem_sh_l2;
  Char_t  has_gem_dg_l1, has_gem_dg_l2;
  Char_t  has_gem_pad_l1, has_gem_pad_l2;
  Float_t gem_trk_eta, gem_trk_phi, gem_trk_rho;
};

class GEMDigiAnalyzer : public edm::EDAnalyzer
{
public:
  /// constructor
  explicit GEMDigiAnalyzer(const edm::ParameterSet&);
  /// destructor
  ~GEMDigiAnalyzer();

  virtual void beginRun(edm::Run const&, edm::EventSetup const&);

  virtual void beginJob() ;

  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  virtual void endJob() ;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:

  void bookGEMDigiTree();
  void bookGEMCSCPadDigiTree();
  void bookGEMCSCCoPadDigiTree();
  void bookSimTracksTree();

  void analyzeGEM();
  void analyzeGEMCSCPad();
  void analyzeGEMCSCCoPad();
  bool isSimTrackGood(const SimTrack &);
  void analyzeTracks(edm::ParameterSet, const edm::Event&, const edm::EventSetup&);
  void buildLUT();
  std::pair<int,int> getClosestChambers(int region, float phi);

  TTree* gem_tree_;
  TTree* gemcscpad_tree_;
  TTree* gemcsccopad_tree_;
  TTree* track_tree_;

  edm::Handle<GEMDigiCollection> gem_digis;
  edm::Handle<GEMCSCPadDigiCollection> gemcscpad_digis;
  edm::Handle<GEMCSCPadDigiCollection> gemcsccopad_digis;
  edm::Handle<edm::SimTrackContainer> sim_tracks;
  edm::Handle<edm::SimVertexContainer> sim_vertices;
  edm::ESHandle<GEMGeometry> gem_geo_;

  const GEMGeometry* gem_geometry_;

  edm::InputTag input_tag_gem_;
  edm::InputTag input_tag_gemcscpad_;
  edm::InputTag input_tag_gemcsccopad_;

  MyGEMDigi gem_digi_;
  MyGEMCSCPadDigis gemcscpad_digi_;
  MyGEMCSCCoPadDigis gemcsccopad_digi_;
  MySimTrack track_;

  edm::ParameterSet cfg_;
  std::string simInputLabel_;
  float minPt_;
  int verbose_;
  float radiusCenter_;
  float chamberHeight_;

  std::pair<std::vector<float>,std::vector<int> > positiveLUT_;
  std::pair<std::vector<float>,std::vector<int> > negativeLUT_;
};
