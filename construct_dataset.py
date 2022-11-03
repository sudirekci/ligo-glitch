import waveform_dataset_3p as wd

directory = '/home/su.direkci/glitch_project/dataset_no_glitch_3p_svd_100_extrinsic_4/'
path_to_glitschen = '/home/su.direkci/programs/glitschen'

add_glitch = True

training_wg = wd.WaveformGenerator(dataset_len=20000, add_glitch=add_glitch, add_noise=False,
                                   directory=directory, path_to_glitschen=path_to_glitschen,
                                   svd_no_basis_coeffs=100, extrinsic_at_train=True)
print(training_wg.priors)
training_wg.construct_signal_dataset(perform_svd=True, save=False)
#training_wg.normalize_params()
#training_wg.normalize_dataset()

training_wg.save_data('training_data')

print('training data saved')

# validation data
validation_wg = wd.WaveformGenerator(dataset_len=2000, add_glitch=add_glitch, add_noise=False,
                                     directory=directory, path_to_glitschen=path_to_glitschen,
                                     svd_no_basis_coeffs=100, extrinsic_at_train=True)
validation_wg.construct_signal_dataset(perform_svd=False)
validation_wg.perform_svd(training_wg.svd.Vh)
validation_wg.calculate_dataset_statistics()

#validation_wg.normalize_params()
#validation_wg.normalize_dataset()

validation_wg.save_data('validation_data')

print('validation data saved')

testing_wg = wd.WaveformGenerator(dataset_len=2000, add_glitch=add_glitch, add_noise=False,
                                  directory=directory, path_to_glitschen=path_to_glitschen,
                                  svd_no_basis_coeffs=100, extrinsic_at_train=True)
testing_wg.construct_signal_dataset(perform_svd=False, save=True, filename='testing_data')

print('testing data saved')
