import h5py
import numpy as np
import os

mock_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'mock')
os.makedirs(mock_dir, exist_ok=True)

N_TRAJ  = 2
N_TIME  = 20
N_SPACE = 256

alpha_values = [-1.0, -2.0, -3.0, -4.0, -5.0]
zeta_values  = [1.0, 3.0]

print("Creating mock dataset...")

for zeta in zeta_values:
    for alpha in alpha_values:
        fname = f"active_matter_L_10.0_zeta_{zeta}_alpha_{alpha}.hdf5"
        fpath = os.path.join(mock_dir, fname)

        with h5py.File(fpath, 'w') as f:
            scalars = f.create_group('scalars')
            scalars.create_dataset('alpha', data=np.float32(alpha))
            scalars.create_dataset('zeta',  data=np.float32(zeta))
            scalars.create_dataset('L',     data=np.float32(10.0))

            dims = f.create_group('dimensions')
            dims.create_dataset('time', data=np.linspace(0, 1, N_TIME).astype(np.float32))
            dims.create_dataset('x',    data=np.linspace(0, 1, N_SPACE).astype(np.float32))
            dims.create_dataset('y',    data=np.linspace(0, 1, N_SPACE).astype(np.float32))

            bc = f.create_group('boundary_conditions')
            bc.create_group('x_periodic').create_dataset('mask', data=np.ones(N_SPACE, dtype=bool))
            bc.create_group('y_periodic').create_dataset('mask', data=np.ones(N_SPACE, dtype=bool))

            t0 = f.create_group('t0_fields')
            t0.create_dataset('concentration',
                data=np.random.randn(N_TRAJ, N_TIME, N_SPACE, N_SPACE).astype(np.float32))

            t1 = f.create_group('t1_fields')
            t1.create_dataset('velocity',
                data=np.random.randn(N_TRAJ, N_TIME, N_SPACE, N_SPACE, 2).astype(np.float32))

            t2 = f.create_group('t2_fields')
            t2.create_dataset('D',
                data=np.random.randn(N_TRAJ, N_TIME, N_SPACE, N_SPACE, 2, 2).astype(np.float32))
            t2.create_dataset('E',
                data=np.random.randn(N_TRAJ, N_TIME, N_SPACE, N_SPACE, 2, 2).astype(np.float32))

        print(f"  created: {fname}")

print(f"\nDone! {len(alpha_values) * len(zeta_values)} mock files in {mock_dir}")
print("Note: these files are gitignored and will not be committed to the repo.")
print("Run this script any time you need to regenerate the mock data.")
