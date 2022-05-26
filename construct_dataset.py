import waveform_dataset as wd

training_wg = wd.WaveformGenerator(dataset_len=1000, add_glitch=True)
training_wg.construct_signal_dataset(perform_svd=True, save=True, filename='training_data')



# validation data
validation_wg = wd.WaveformGenerator(dataset_len=100, add_glitch=True)
validation_wg.construct_signal_dataset(perform_svd=False)
validation_wg.perform_svd(training_wg.svd.Vh)
validation_wg.calculate_dataset_statistics()

validation_wg.save_data('validation_data')

testing_wg = wd.WaveformGenerator(dataset_len=100, add_glitch=True)
testing_wg.construct_signal_dataset(perform_svd=False, save=True, filename='testing_data')

