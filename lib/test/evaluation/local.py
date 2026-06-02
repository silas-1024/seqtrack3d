from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/media/pc-4090/9bd4d9e0-148f-4dd1-a9c6-1f4c9dff8e4e/datasets/got10k_lmdb'
    settings.got10k_path = '/media/pc-4090/9bd4d9e0-148f-4dd1-a9c6-1f4c9dff8e4e/datasets/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_extension_subset_path = '/media/pc-4090/9bd4d9e0-148f-4dd1-a9c6-1f4c9dff8e4e/datasets/lasot_extension_subset'
    settings.lasot_lmdb_path = '/media/pc-4090/9bd4d9e0-148f-4dd1-a9c6-1f4c9dff8e4e/datasets/lasot_lmdb'
    settings.lasot_path = '/media/pc-4090/9bd4d9e0-148f-4dd1-a9c6-1f4c9dff8e4e/datasets/lasot'
    settings.lasotlang_path = '/media/pc-4090/9bd4d9e0-148f-4dd1-a9c6-1f4c9dff8e4e/datasets/lasot'
    settings.network_path = '/media/lisuran/seqtrack3d/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/media/pc-4090/9bd4d9e0-148f-4dd1-a9c6-1f4c9dff8e4e/datasets/nfs'
    settings.otb_path = '/media/pc-4090/9bd4d9e0-148f-4dd1-a9c6-1f4c9dff8e4e/datasets/OTB2015'
    settings.otblang_path = '/media/pc-4090/9bd4d9e0-148f-4dd1-a9c6-1f4c9dff8e4e/datasets/otb_lang'
    settings.prj_dir = '/media/lisuran/seqtrack3d'
    settings.result_plot_path = '/media/lisuran/seqtrack3d/output/test/result_plots'
    settings.results_path = '/media/lisuran/seqtrack3d/output/test/tracking_results'    # Where to store tracking results
    settings.satsot_path = '/media/pc-4090/9bd4d9e0-148f-4dd1-a9c6-1f4c9dff8e4e/datasets/satsot'
    settings.save_dir = '/media/lisuran/seqtrack3d/output'
    settings.segmentation_path = '/media/lisuran/seqtrack3d/output/test/segmentation_results'
    settings.sv248s_test_path = '/media/pc-4090/9bd4d9e0-148f-4dd1-a9c6-1f4c9dff8e4e/datasets/sv248/test_sv'
    settings.tc128_path = '/media/pc-4090/9bd4d9e0-148f-4dd1-a9c6-1f4c9dff8e4e/datasets/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/media/pc-4090/9bd4d9e0-148f-4dd1-a9c6-1f4c9dff8e4e/datasets/tnl2k/test'
    settings.tpl_path = ''
    settings.trackingnet_path = '/media/pc-4090/9bd4d9e0-148f-4dd1-a9c6-1f4c9dff8e4e/datasets/trackingnet'
    settings.uav_path = '/media/pc-4090/9bd4d9e0-148f-4dd1-a9c6-1f4c9dff8e4e/datasets/UAV123'
    settings.viso_path = '/media/pc-4090/9bd4d9e0-148f-4dd1-a9c6-1f4c9dff8e4e/datasets/viso'
    settings.vot_path = '/media/pc-4090/9bd4d9e0-148f-4dd1-a9c6-1f4c9dff8e4e/datasets/VOT2019'
    settings.youtubevos_dir = ''

    return settings

