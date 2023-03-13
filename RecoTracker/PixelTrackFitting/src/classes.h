#include "RecoTracker/PixelTrackFitting/interface/PixelFitter.h"
#include "RecoTracker/PixelTrackFitting/interface/PixelTrackFilter.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace RecoPixelVertexing_PixelTrackFitting {
  struct dictionary {
    PixelFitter pf;
    edm::Wrapper<PixelFitter> wpf;

    PixelTrackFilter ptf;
    edm::Wrapper<PixelTrackFilter> wptf;
  };
}  // namespace RecoPixelVertexing_PixelTrackFitting
