#include "VertexBeamspotOrigins.h"

VertexBeamspotOrigins::VertexBeamspotOrigins(const edm::ParameterSet& regPSet, edm::ConsumesCollector& iC) {
  // operation mode
  std::string operationModeString = regPSet.getParameter<std::string>("operationMode");
  if (operationModeString == "BeamSpotFixed")
    m_operationMode = OperationMode::BEAM_SPOT_FIXED;
  else if (operationModeString == "BeamSpotSigma")
    m_operationMode = OperationMode::BEAM_SPOT_SIGMA;
  else if (operationModeString == "VerticesFixed")
    m_operationMode = OperationMode::VERTICES_FIXED;
  else if (operationModeString == "VerticesSigma")
    m_operationMode = OperationMode::VERTICES_SIGMA;
  else
    throw cms::Exception("Configuration") << "Unknown operation mode string: " << operationModeString;

  token_beamSpot = iC.consumes<reco::BeamSpot>(regPSet.getParameter<edm::InputTag>("beamSpot"));
  m_maxNVertices = 1;
  if (m_operationMode == OperationMode::VERTICES_FIXED || m_operationMode == OperationMode::VERTICES_SIGMA) {
    token_vertex = iC.consumes<reco::VertexCollection>(regPSet.getParameter<edm::InputTag>("vertexCollection"));
    m_maxNVertices = regPSet.getParameter<int>("maxNVertices");
  }

  // mode-dependent z-halflength of tracking regions
  m_zErrorBeamSpot = regPSet.getParameter<double>("zErrorBeamSpot");
  if (m_operationMode == OperationMode::VERTICES_SIGMA)
    m_nSigmaZVertex = regPSet.getParameter<double>("nSigmaZVertex");
  if (m_operationMode == OperationMode::VERTICES_FIXED)
    m_zErrorVertex = regPSet.getParameter<double>("zErrorVertex");
  m_nSigmaZBeamSpot = -1.;
  if (m_operationMode == OperationMode::BEAM_SPOT_SIGMA) {
    m_nSigmaZBeamSpot = regPSet.getParameter<double>("nSigmaZBeamSpot");
    if (m_nSigmaZBeamSpot < 0.)
      throw cms::Exception("Configuration") << "nSigmaZBeamSpot must be positive for BeamSpotSigma mode!";
  }
}

void VertexBeamspotOrigins::fillDescriptions(edm::ParameterSetDescription& desc,
                                             const std::string& defaultBeamSpot,
                                             const std::string& defaultVertex,
                                             int defaultMaxVertices) {
  desc.add<std::string>("operationMode", "BeamSpotFixed");
  desc.add<edm::InputTag>("beamSpot", defaultBeamSpot);
  desc.add<edm::InputTag>("vertexCollection", defaultVertex);
  desc.add<int>("maxNVertices", defaultMaxVertices);

  desc.add<double>("nSigmaZBeamSpot", 4.);
  desc.add<double>("zErrorBeamSpot", 24.2);
  desc.add<double>("nSigmaZVertex", 3.);
  desc.add<double>("zErrorVertex", 0.2);
}

VertexBeamspotOrigins::Origins VertexBeamspotOrigins::origins(const edm::Event& iEvent) const {
  Origins ret;

  // always need the beam spot (as a fall back strategy for vertex modes)
  edm::Handle<reco::BeamSpot> bs;
  iEvent.getByToken(token_beamSpot, bs);
  if (!bs.isValid())
    return ret;

  // this is a default origin for all modes
  GlobalPoint default_origin(bs->x0(), bs->y0(), bs->z0());

  // fill the origins and halfLengths depending on the mode
  if (m_operationMode == OperationMode::BEAM_SPOT_FIXED || m_operationMode == OperationMode::BEAM_SPOT_SIGMA) {
    ret.emplace_back(
        default_origin,
        (m_operationMode == OperationMode::BEAM_SPOT_FIXED) ? m_zErrorBeamSpot : m_nSigmaZBeamSpot * bs->sigmaZ());
  } else if (m_operationMode == OperationMode::VERTICES_FIXED || m_operationMode == OperationMode::VERTICES_SIGMA) {
    edm::Handle<reco::VertexCollection> vertices;
    iEvent.getByToken(token_vertex, vertices);
    int n_vert = 0;
    for (const auto& v : *vertices) {
      if (v.isFake() || !v.isValid())
        continue;

      ret.emplace_back(
          GlobalPoint(v.x(), v.y(), v.z()),
          (m_operationMode == OperationMode::VERTICES_FIXED) ? m_zErrorVertex : m_nSigmaZVertex * v.zError());
      ++n_vert;
      if (m_maxNVertices >= 0 && n_vert >= m_maxNVertices) {
        break;
      }
    }
    // no-vertex fall-back case:
    if (ret.empty()) {
      ret.emplace_back(default_origin, (m_nSigmaZBeamSpot > 0.) ? m_nSigmaZBeamSpot * bs->z0Error() : m_zErrorBeamSpot);
    }
  }

  return ret;
}
