"""Reproduce the authored DRAFT library in implementation and spec directories."""
from pathlib import Path
from .common import write


def build():
    draft = {'status': 'draft', 'reviewer': None, 'reviewed_at': None, 'evidence': None}
    def concept(cid, stream, desc, channels, unknown):
        from .common import STREAMS
        return dict(id=cid, stream=stream, family=STREAMS[stream], description=desc,
                    channels=channels, scope={'countries':['Argentina','Brazil','Germany','Uruguay'],
                    'crops':['corn','rapeseed','soybean','wheat'],'activation':'only expert-approved strata'},
                    units_provenance='dataset/yieldsat_schema.py PROVENANCE; physical units are not presumed audited',
                    reference_population='must be defined and fitted on this fold training geography before numeric activation',
                    support='one observed 64x64 tile; coarse source support retained',
                    window='declared eligible observation window; retrospective or days after seeding',
                    unknown_if=unknown, review=dict(draft))
    concepts = [
        concept('rain_supported', 'weather', 'A crop-stage interval with precipitation sufficient to ease water limitation relative to a reviewed local seasonal reference.', ['total_prec'], ['irrigation or water balance unknown', 'first weather interval', 'reference or interval conversion unreviewed']),
        concept('optical_growth', 'series', 'An increase in canopy vegetation optical response during the reviewed crop growth window, supported by the dated red and near-infrared series.', ['B04','B08'], ['cloud/shadow contamination', 'fewer than two valid dates', 'crop stage not established']),
        concept('warm_regime', 'weather', 'A warmer observed temperature regime relative to a reviewed crop- and season-specific reference, within that crop\'s non-stress range.', ['temp_mean','temp_max','temp_min'], ['Kelvin-day interval semantics unverified', 'crop thermal range unknown']),
        concept('active_spectral_state', 'spec', 'A red-edge and near-infrared spectral state consistent with active vegetation in the reviewed crop stage.', ['B05','B06','B07','B08','B8A'], ['cloud/shadow contamination', 'band scaling uncertain', 'phenological context missing']),
        concept('clay_rich_surface', 'soil', 'Relatively clay-rich mineral soil in the upper 0–30 cm against a reviewed local soil reference.', ['clay_0-5','clay_5-15','clay_15-30'], ['soil scale/depth mapping unverified', 'reference unknown']),
        concept('persistent_canopy', 'series', 'A relatively persistent canopy optical signal through a reviewed dry interval.', ['B04','B08','B11','B12'], ['dry interval not established', 'irrigation/management unknown', 'too few valid optical dates']),
        concept('low_elevation_position', 'dem', 'A relatively low elevation position within a verified local landscape reference, not globally low altitude.', ['dem'], ['landscape reference unavailable', 'DEM units or datum unverified']),
        concept('fine_surface_texture', 'soil', 'Relatively fine-textured surface soil in the reviewed 0–30 cm depth range.', ['clay_0-5','silt_0-5','sand_0-5','clay_5-15','clay_15-30'], ['depth aggregation unreviewed', 'soil interpolation dominates local variation']),
        concept('pole_facing_aspect', 'terrain', 'A terrain aspect facing the local pole in a non-flat, correctly geolocated landscape.', ['aspect'], ['aspect convention unverified', 'flat terrain or slope validity unknown', 'hemisphere not established']),
        concept('cool_regime', 'weather', 'A relatively cool observed temperature regime in the reviewed seasonal reference.', ['temp_mean'], ['interval conversion unverified', 'local reference unavailable']),
        concept('organic_surface', 'soil', 'Relatively organic-carbon-rich mineral topsoil in the reviewed upper 0–30 cm profile.', ['soc_0-5','soc_5-15','soc_15-30'], ['soil units or reference unverified', 'organic/mineral soil context unknown']),
        concept('spectral_moisture_state', 'spec', 'A near-infrared/SWIR state consistent with relatively moist vegetation under a reviewed crop-specific interpretation.', ['B08','B8A','B11','B12'], ['soil background/clouds confound reflectance', 'interpretation unvalidated']),
    ]
    pairs = [
        ('rain_supported','optical_growth','Sufficient precipitation may support a later canopy response during active growth when water limitation and other required conditions are established.'),
        ('warm_regime','active_spectral_state','Within a crop-specific non-stress thermal range, warmer growing conditions may accompany an active vegetation spectral state.'),
        ('clay_rich_surface','persistent_canopy','Clay-rich soil may be associated with canopy persistence during limited rainfall under an appropriate soil and management context; waterlogging and rooting constraints can reverse the association.'),
        ('low_elevation_position','fine_surface_texture','In a verified depositional landscape, relatively low positions may be associated with finer surface texture; this is not a universal altitude–soil rule.'),
        ('pole_facing_aspect','cool_regime','At matching local spatial support, pole-facing terrain may have a cooler thermal regime; field-centroid ERA5 may be unable to resolve this and must then abstain.'),
        ('organic_surface','spectral_moisture_state','In a reviewed mineral-soil and crop context, organic-carbon-rich surface soil may accompany a moist vegetation spectral state; management and other water controls remain necessary qualifications.'),
    ]
    rules = [dict(id='ys_r%02d'%(i+1), concept_a=a, concept_b=b, statement=s,
                  direction='a_to_b', relation_type='qualified association',
                  scope='only expert-approved geography/crop/time strata; no universal claim',
                  window='same declared window for both endpoints; any within-window lag requires observable dates',
                  scorer='directed_displacement_v1',
                  qualifications=['association only; not causation', 'expert review of region/crop and numeric interpretation required'],
                  required_context=['reviewed physical units and source support', 'observable crop/season conditions', 'all statement-specific qualifications supported'],
                  minimum_presence=0.5, review=dict(draft)) for i,(a,b,s) in enumerate(pairs)]
    return dict(schema_version=1, version='yieldsat_image_draft_2026-10-02',
                generator={'kind':'assistant-authored seed drafts', 'external_api_used':False},
                concepts=concepts, rules=rules)


def render(lib):
    lines=['# YieldSAT concept and relationship drafts','',
           'All entries are **drafts requiring human expert review**. No agronomic approval or validated geometry is claimed.', '',
           'Unobservable qualifications require abstention. These candidates deliberately include difficult cases for the pilot; none is automatically activated.', '',
           '## Concepts','', '| ID | Stream | Description |', '|---|---|---|']
    lines += ['| %s | %s | %s |'%(c['id'],c['stream'],c['description']) for c in lib['concepts']]
    lines += ['', '## Rules', '']
    for r in lib['rules']:
        lines += ['### '+r['id'], '', r['concept_a']+' → '+r['concept_b'], '', r['statement'], '']
    return '\n'.join(lines)


def main():
    root=Path(__file__).resolve().parent
    lib=build()
    for dest in (root/'assets',root.parent/'spec'/'yieldsat_knowledge_assets'):
        write(dest/'library.json',lib)
        (dest/'library.md').write_text(render(lib))

if __name__=='__main__': main()
