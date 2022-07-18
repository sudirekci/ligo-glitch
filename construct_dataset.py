import waveform_dataset as wd

directory = '/home/su.direkci/glitch_project/glitch_dataset/'
path_to_glitschen = '/home/su.direkci/programs/glitschen/glitschen'

training_wg = wd.WaveformGenerator(dataset_len=1000, add_glitch=True,
                                   directory=directory, path_to_glitschen=path_to_glitschen)
training_wg.construct_signal_dataset(perform_svd=True, save=True, filename='training_data')

# validation data
validation_wg = wd.WaveformGenerator(dataset_len=100, add_glitch=True,
                                     directory=directory, path_to_glitschen=path_to_glitschen)
validation_wg.construct_signal_dataset(perform_svd=False)
validation_wg.perform_svd(training_wg.svd.Vh)
validation_wg.calculate_dataset_statistics()

validation_wg.save_data('validation_data')

testing_wg = wd.WaveformGenerator(dataset_len=100, add_glitch=True,
                                  directory=directory, path_to_glitschen=path_to_glitschen)
testing_wg.construct_signal_dataset(perform_svd=False, save=True, filename='testing_data')
