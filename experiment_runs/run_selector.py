# from experiment_runs.configs import debug, cremad, tmp_new
from experiment_runs.configs import tmp_new, pcgan, whisper, whisper_pcgan, melgan_finetuning, filter_wgan

AVAILABLE_RUNS = {
    # 'debug': debug.Q,
    # 'cremad': cremad.Q,
    'tmp_new': tmp_new.Q,
    'pcgan': pcgan.Q,
    'whisper': whisper.Q,
    'whisper-pcgan': whisper_pcgan.Q,
    'melgan_finetune': melgan_finetuning.Q,
    'filter-wgan': filter_wgan.Q,
}