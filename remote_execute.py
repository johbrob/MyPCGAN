# import sanity_checks.audio_mel_conversion
import whisper_gender_classification
from sanity_checks.tmp_run import compare_melgan_version, whisper_as_melgan
import sanity_checks.compare_spectrograms

if __name__ == '__main__':
    # sanity_checks.audio_mel_conversion.main()
    # whisper_gender_classification.main()
    # compare_melgan_version()
    # whisper_as_melgan()
    sanity_checks.compare_spectrograms.main()