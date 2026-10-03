# Vertex-slot GNN primary-vertex reconstruction (Alpaka backend)

Primary-vertex reconstruction for Phase-2 with MTD timing using the *vertex-slot* model: a
graph/attention network that assigns every selected track to one of `K` vertex slots and predicts
for every slot its z position, time and existence probability. The model runs through
`PhysicsTools/PyTorchAlpaka` (GPU, or CPU with the serial Alpaka backend); the vertices are built and
fitted by the standard `PrimaryVertexProducer`.

## Modules (RecoVertex/PrimaryVertexProducer)

| module | type | what it does |
|---|---|---|
| `trackFeatureProducer` | `vertexgnn::TrackFeatureProducer` (plugins/) | selects the tracks with the `TrackFilterForPVFinding` of the PV producer and fills the model inputs, one SoA row per track (`DataFormats/VertexGNNReco`, `TrackFeaturesLayout`) |
| `gnnVertexProducer` | `vertexgnn::GNNVertexProducerAlpaka@alpaka` (plugins/alpaka/) | runs the TorchScript model on the feature SoA, output SoA `GNNOutputLayout`: `A[N,K]` assignment probabilities, `z_hat`, `t_hat`, `p` per slot (replicated per row), `pi[N,3]` PID weights |
| `unsortedOfflinePrimaryVerticesGNN` | `PrimaryVertexProducer`, algorithm `GNN2D_alpaka` | per track the slot with the highest probability; slots with `p > existenceThreshold` and at least one track become vertex candidates at `z_hat`, tracks join their slot if the probability is at least `trackAssignmentThreshold` (`GNNClusterizerFromAlpaka`); candidates are fitted with the configured fitter (`useClusterWeights` keeps the GNN probabilities as track weights instead of the fitter's); per-track outputs stored as ValueMaps `gnnSlotAssignment`, `gnnMaxProb`, `gnnPiWeight0/1/2` |
| `offlinePrimaryVerticesGNN`, `tofPIDGNN` | standard sorting and tofPID | as for the 4D vertices |
| `GNNTrackInspector` | analyzer (test/ configs) | histograms of the per-track ValueMaps |

The number of slots `K` is the compile-time constant `vertexgnn::kNumSlots`
(`DataFormats/VertexGNNReco/interface/VertexGNNSoA.h`) and the model input features are the `kNumFeatures`
columns of `TrackFeaturesLayout` in the same header, in that order. **A model can only be used with a build whose
`kNumSlots` and feature list match it.**

## Model file

`RecoVertex/PrimaryVertexProducer/data/vertexSlotGNN.pt` (cms-data): TorchScript module that takes the
`[N, kNumFeatures]` feature tensor and returns the five tensors `A[N,K]`, `z_hat[N,K]`, `t_hat[N,K]`,
`p[N,K]`, `pi[N,3]`, every output with one row per track. `GNNVertexProducerAlpaka` runs the model once
at construction on random input and stops with a configuration error when the model does not accept
`kNumFeatures` features or does not return `kNumSlots` slots.

## Configuration

The vertex-slot vertices are added to the Phase-2 (timing layer) vertex reconstruction by the
process modifier `vertexSlotGNN` (`Configuration/ProcessModifiers/python/vertexSlotGNN_cff.py`),
see `RecoVertex/Configuration/python/RecoVertex_phase2_timing_cff.py` for the module definitions and
`RecoVertex_cff.py` / `RecoVertex_EventContent_cff.py` for the task and the event content. Alpaka
modules need the `ProcessAcceleratorAlpaka` service (`--procModifiers alpaka` with `cmsDriver`, or an
explicit `process.load` as in the test configuration).

`test/vertexTask_alpaka_cfg.py` re-runs the vertex reconstruction (3D, 4D and GNN) on a Phase-2
GEN-SIM-RECO file; `test/mtdValidation_3way_cfg.py` and `test/mtdHarvesting_3way_cfg.py` run the MTD
validation and harvesting for the three vertex collections into the DQM folders `MTD/Vertices/{3D,4D,GNN}`
and `MTD/Tracks/{3D,4D,GNN}`. When merging several validation files, harvest each file first and
`hadd` the harvested outputs.
