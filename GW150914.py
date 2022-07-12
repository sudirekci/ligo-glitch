import bilby
import h5py
import numpy as np
import argparse
from gwpe import Nestedspace
from gwpe import PosteriorModel

"""

python -m GW150914 \
    --data_dir /home/su/Documents/normalizing_flows/data/events/GW150914/ \
    --model_dir /home/su/Documents/normalizing_flows/models/GW150914/ \
    --out_dir /home/su/Documents/normalizing_flows/bilby_runs/GW150914/
    
"""


def parse_args():

    #parser = argparse.ArgumentParser()

    dir_parent_parser = argparse.ArgumentParser(add_help=False)
    dir_parent_parser.add_argument('--data_dir', type=str, required=True)
    dir_parent_parser.add_argument('--model_dir', type=str, required=True)
    dir_parent_parser.add_argument('--no_cuda', action='store_false', dest='cuda')
    dir_parent_parser.add_argument('--out_dir', type=str, required=True)

    ns = Nestedspace()

    return dir_parent_parser.parse_args(namespace=ns)



def main():

    args = parse_args()

    print('Waveform directory', args.data_dir)
    print('Model directory', args.model_dir)
    pm = PosteriorModel(model_dir=args.model_dir,
                        data_dir=args.data_dir,
                        use_cuda=args.cuda)
    print('Device', pm.device)


    outdir = args.out_dir
    label = 'GW150914'

    event_strain = {}
    with h5py.File(args.data_dir+'strain_FD_whitened.hdf5', 'r') as f:
        event_strain['H1'] = f['H1'][:].astype(np.complex64)
        event_strain['L1'] = f['L1'][:].astype(np.complex64)

    # Load bilby samples
    result = bilby.result.read_in_result(outdir=outdir, label=label)

    bilby_samples = result.posterior[['mass_1', 'mass_2', 'phase', 'geocent_time', 'luminosity_distance',
                                      'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl',
                                      'theta_jn', 'psi', 'ra', 'dec']].values

    # Shift the time of coalescence by the trigger time
    bilby_samples[:,3] = bilby_samples[:,3] - pm.wfd.ref_time


if __name__ == "__main__":
    main()