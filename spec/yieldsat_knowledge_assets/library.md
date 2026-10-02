# YieldSAT concept and relationship drafts

All entries are **drafts requiring human expert review**. No agronomic approval or validated geometry is claimed.

Unobservable qualifications require abstention. These candidates deliberately include difficult cases for the pilot; none is automatically activated.

## Concepts

| ID | Stream | Description |
|---|---|---|
| rain_supported | weather | A crop-stage interval with precipitation sufficient to ease water limitation relative to a reviewed local seasonal reference. |
| optical_growth | series | An increase in canopy vegetation optical response during the reviewed crop growth window, supported by the dated red and near-infrared series. |
| warm_regime | weather | A warmer observed temperature regime relative to a reviewed crop- and season-specific reference, within that crop's non-stress range. |
| active_spectral_state | spec | A red-edge and near-infrared spectral state consistent with active vegetation in the reviewed crop stage. |
| clay_rich_surface | soil | Relatively clay-rich mineral soil in the upper 0–30 cm against a reviewed local soil reference. |
| persistent_canopy | series | A relatively persistent canopy optical signal through a reviewed dry interval. |
| low_elevation_position | dem | A relatively low elevation position within a verified local landscape reference, not globally low altitude. |
| fine_surface_texture | soil | Relatively fine-textured surface soil in the reviewed 0–30 cm depth range. |
| pole_facing_aspect | terrain | A terrain aspect facing the local pole in a non-flat, correctly geolocated landscape. |
| cool_regime | weather | A relatively cool observed temperature regime in the reviewed seasonal reference. |
| organic_surface | soil | Relatively organic-carbon-rich mineral topsoil in the reviewed upper 0–30 cm profile. |
| spectral_moisture_state | spec | A near-infrared/SWIR state consistent with relatively moist vegetation under a reviewed crop-specific interpretation. |

## Rules

### ys_r01

rain_supported → optical_growth

Sufficient precipitation may support a later canopy response during active growth when water limitation and other required conditions are established.

### ys_r02

warm_regime → active_spectral_state

Within a crop-specific non-stress thermal range, warmer growing conditions may accompany an active vegetation spectral state.

### ys_r03

clay_rich_surface → persistent_canopy

Clay-rich soil may be associated with canopy persistence during limited rainfall under an appropriate soil and management context; waterlogging and rooting constraints can reverse the association.

### ys_r04

low_elevation_position → fine_surface_texture

In a verified depositional landscape, relatively low positions may be associated with finer surface texture; this is not a universal altitude–soil rule.

### ys_r05

pole_facing_aspect → cool_regime

At matching local spatial support, pole-facing terrain may have a cooler thermal regime; field-centroid ERA5 may be unable to resolve this and must then abstain.

### ys_r06

organic_surface → spectral_moisture_state

In a reviewed mineral-soil and crop context, organic-carbon-rich surface soil may accompany a moist vegetation spectral state; management and other water controls remain necessary qualifications.
