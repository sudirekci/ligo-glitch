import waveform_dataset_3p as wd

directory = '/home/su.direkci/glitch_project/dataset_no_glitch_3p_svd_10_extrinsic/'
path_to_glitschen = '/home/su.direkci/programs/glitschen'

training_wg = wd.WaveformGenerator(dataset_len=500000, add_glitch=False, add_noise=False,
                                   directory=directory, path_to_glitschen=path_to_glitschen,
                                   svd_no_basis_coeffs=10)
print(training_wg.priors)
training_wg.construct_signal_dataset(perform_svd=True, save=True, filename='training_data')
training_wg.normalize_params()
training_wg.normalize_dataset()

print('training data saved')

# validation data
validation_wg = wd.WaveformGenerator(dataset_len=50000, add_glitch=False, add_noise=False,
                                     directory=directory, path_to_glitschen=path_to_glitschen,
                                     svd_no_basis_coeffs=10)
validation_wg.construct_signal_dataset(perform_svd=False)
validation_wg.perform_svd(training_wg.svd.Vh)
validation_wg.calculate_dataset_statistics()

validation_wg.normalize_params()
validation_wg.normalize_dataset()

validation_wg.save_data('validation_data')

print('validation data saved')

testing_wg = wd.WaveformGenerator(dataset_len=50000, add_glitch=False, add_noise=False,
                                  directory=directory, path_to_glitschen=path_to_glitschen,
                                  svd_no_basis_coeffs=10)
testing_wg.construct_signal_dataset(perform_svd=False, save=True, filename='testing_data')


print('testing data saved')
