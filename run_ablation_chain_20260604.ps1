Set-Location 'e:\cydd_own_products\memorax-test-locomo-enrichment-append-merge-mmwrite-audit\evaluation_pipeline\v3_ablate_v4add'
python run.py --config config.conv1_v3add_v4search.yaml --from search
Set-Location 'e:\cydd_own_products\memorax-test-locomo-enrichment-append-merge-mmwrite-audit\evaluation_pipeline\v4_ablate_nosearchslot'
python run.py --config config.conv1_v4_noslot.yaml
