'''
demo: fit FLAME face model to 3D landmarks
Tianye Li <tianye.li@tuebingen.mpg.de>
'''

from __future__ import print_function

import argparse
import numpy as np
import chumpy as ch
from os import listdir
from os.path import join, isfile, isdir, splitext, dirname, abspath

from smpl_webuser.serialization import load_model
from fitting.landmarks import load_embedding, landmark_error_3d
from fitting.util import load_binary_pickle, write_simple_obj, safe_mkdir

import re


# -----------------------------------------------------------------------------

# Do not set in arg parse as default, because we load only csv's from the args, not pkl files
DEFAULT_LMK_PATH = './data/landmark_3d.pkl'

# Fixme: Number of Feature point and Dimension are hardcoded
N_FEATS = 49
N_FEAT_DIM = 3

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="FLAME Model fitting.")
    parser.add_argument("--lmks", type=str,
                        help="Path to 3D landmarks file/dir, in csv format. Default uses a test file. "
                             "Directories will generate a fitted model for each landmark file.")
    parser.add_argument("--average", action='store_true',
                        help="Will average a set of landmarks in a directory")
    parser.add_argument("--preserve-shape", action='store_true',
                        help="Will fit shape on the first landmarks in a dir, "
                             "the remainder will only fit expression and pose")
    return parser.parse_args()

# -----------------------------------------------------------------------------

def natural_sort(l, reverse=False):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key, reverse=reverse)

def load_lmk_file(path):
    return np.genfromtxt(path, delimiter=',')


def load_lmks_dir(path, average=False):
    # Preprocess directory looking for landmark files
    file_list = [filename for filename in natural_sort(listdir(path)) if filename.endswith(".csv")]
    file_list = list(map(lambda f: join(path, f), file_list))

    if average is False:
        # Load many landmarks to fit model for each
        lmk_3d = list(map(load_lmk_file, file_list))
    else:
        # Load many landmarks to average
        print("Loading from Directory...")
        lmk_3d = np.zeros([N_FEATS, N_FEAT_DIM])
        for filename in file_list:
            lmk_3d_t = load_lmk_file(filename)
            lmk_3d = np.add(lmk_3d, lmk_3d_t)

        lmk_3d = [np.divide(lmk_3d, len(file_list))]

    return lmk_3d, file_list

'''
# Load Landmarks loads a landmark file or files from a director
'''
def load_lmks(path=None, average=False):
    lmks_3d, files = ([], [])
    if path is None:
        path = DEFAULT_LMK_PATH
        lmks_3d = load_binary_pickle(DEFAULT_LMK_PATH)
        lmks_3d = [lmks_3d]
        files = [path]
    elif isfile(path):
        lmks_3d = load_lmk_file(path)
        lmks_3d = [lmks_3d]
        files = [path]
    elif isdir(path):
        lmks_3d, files = load_lmks_dir(path, average)
    else:
        print("Invalid File: {}".format(path))
        exit(-1)

    print("loaded 3d landmark from:", path)
    return lmks_3d, files

# -----------------------------------------------------------------------------

def fit_lmk3d( lmk_3d,                      # input landmark 3d
               model,                       # model
               lmk_face_idx, lmk_b_coords,  # landmark embedding
               weights,                     # weights for the objectives
               shape_num=300, expr_num=100, opt_options=None,
               preserve_shape=False):
    """ function: fit FLAME model to 3d landmarks

    input: 
        lmk_3d: input landmark 3d, in shape (N,3)
        model: FLAME face model
        lmk_face_idx, lmk_b_coords: landmark embedding, in face indices and barycentric coordinates
        weights: weights for each objective
        shape_num, expr_num: numbers of shape and expression compoenents used
        opt_options: optimizaton options

    output:
        model.r: fitted result vertices
        model.f: fitted result triangulations (fixed in this code)
        parms: fitted model parameters

    """

    # variables
    shape_idx      = np.arange( 0, min(300,shape_num) )        # valid shape component range in "betas": 0-299
    expr_idx       = np.arange( 300, 300+min(100,expr_num) )   # valid expression component range in "betas": 300-399
    used_idx       = np.union1d( shape_idx, expr_idx ) if preserve_shape is False else expr_idx

    if preserve_shape is False:
        model.betas[:] = np.random.rand( model.betas.size ) * 0.0  # initialized to zero

    model.pose[:]  = np.random.rand( model.pose.size ) * 0.0   # initialized to zero
    free_variables = [ model.trans, model.pose, model.betas[used_idx] ] 
    
    # weights
    print("fit_lmk3d(): use the following weights:")
    for kk in weights.keys():
        print("fit_lmk3d(): weights['%s'] = %f" % ( kk, weights[kk] ))

    # objectives
    # lmk
    lmk_err = landmark_error_3d( mesh_verts=model, 
                                 mesh_faces=model.f, 
                                 lmk_3d=lmk_3d, 
                                 lmk_face_idx=lmk_face_idx, 
                                 lmk_b_coords=lmk_b_coords, 
                                 weight=weights['lmk'] )
    # regularizer
    shape_err = weights['shape'] * model.betas[shape_idx] 
    expr_err  = weights['expr']  * model.betas[expr_idx] 
    pose_err  = weights['pose']  * model.pose[3:] # exclude global rotation
    objectives = {}
    objectives.update( { 'lmk': lmk_err, 'shape': shape_err, 'expr': expr_err, 'pose': pose_err } ) 

    # options
    if opt_options is None:
        print("fit_lmk3d(): no 'opt_options' provided, use default settings.")
        import scipy.sparse as sp
        opt_options = {}
        opt_options['disp']    = 1
        opt_options['delta_0'] = 0.1
        opt_options['e_3']     = 1e-4
        opt_options['maxiter'] = 100
        sparse_solver = lambda A, x: sp.linalg.cg(A, x, maxiter=opt_options['maxiter'])[0]
        opt_options['sparse_solver'] = sparse_solver

    # on_step callback
    def on_step(_):
        pass
        
    # optimize
    # step 1: rigid alignment
    from time import time
    timer_start = time()
    print("\nstep 1: start rigid fitting...")
    ch.minimize( fun      = lmk_err,
                 x0       = [ model.trans, model.pose[0:3] ],
                 method   = 'dogleg',
                 callback = on_step,
                 options  = opt_options )
    timer_end = time()
    print("step 1: fitting done, in %f sec\n" % ( timer_end - timer_start ))

    # step 2: non-rigid alignment
    timer_start = time()
    print("step 2: start non-rigid fitting...")
    ch.minimize( fun      = objectives,
                 x0       = free_variables,
                 method   = 'dogleg',
                 callback = on_step,
                 options  = opt_options )
    timer_end = time()
    print("step 2: fitting done, in %f sec\n" % ( timer_end - timer_start ))

    # return results
    parms = { 'trans': model.trans.r, 'pose': model.pose.r, 'betas': model.betas.r }
    return model.r, model.f, parms

# -----------------------------------------------------------------------------


def run_fitting(lmk_3d, model=None,
                output_file='./output/fit_lmk3d_result.obj',
                preserve_shape=False):

    # model
    if model is None:
        preserve_shape=False
        model_path = './models/male_model.pkl' # change to 'female_model.pkl' or 'generic_model.pkl', if needed
        model = load_model( model_path )       # the loaded model object is a 'chumpy' object, check https://github.com/mattloper/chumpy for details
        print("loaded model from:", model_path)

    # landmark embedding
    lmk_emb_path = './data/lmk_embedding_intraface_to_flame.pkl' 
    lmk_face_idx, lmk_b_coords = load_embedding( lmk_emb_path )
    print("loaded lmk embedding")

    # output
    output_dir = dirname(abspath(output_file))
    safe_mkdir( output_dir )

    # weights
    # TODO: Load from config
    weights = {}
    weights['lmk']   = 1.0
    weights['shape'] = 0.001
    weights['expr']  = 0.001
    weights['pose']  = 0.1
    
    # optimization options
    import scipy.sparse as sp
    opt_options = {}
    opt_options['disp']    = 1
    opt_options['delta_0'] = 0.1
    opt_options['e_3']     = 1e-4
    opt_options['maxiter'] = 100
    sparse_solver = lambda A, x: sp.linalg.cg(A, x, maxiter=opt_options['maxiter'])[0]
    opt_options['sparse_solver'] = sparse_solver

    print("Preserve Shape: ", preserve_shape)

    # run fitting
    mesh_v, mesh_f, parms = fit_lmk3d( lmk_3d=lmk_3d,                                         # input landmark 3d
                                       model=model,                                           # model
                                       lmk_face_idx=lmk_face_idx, lmk_b_coords=lmk_b_coords,  # landmark embedding
                                       weights=weights,                                       # weights for the objectives
                                       shape_num=299, expr_num=100, opt_options=opt_options,  # options
                                       preserve_shape=preserve_shape) # preserves shape

    # write result
    write_simple_obj( mesh_v=mesh_v, mesh_f=mesh_f, filepath=output_file, verbose=False )

    # Return fitted model
    return model, output_file

# -----------------------------------------------------------------------------

def main():
    args = get_arguments()

    # Initialize
    model = None
    lmks_3d, files = load_lmks(path=args.lmks, average=args.average)

    for idx, (lmk, file) in enumerate(zip(lmks_3d, files)):
        print("Landmark: {} File: {}".format(lmk.shape, file))
        output_file=splitext(file)[0] + '.obj'
        print("Out:", output_file)

        # FYI: Giving run_fitting model=None will make it load the model everytime...
        # BUT we get a clean model eachtime!!!
        model = None if not args.preserve_shape else model

        # if model is None, we need to generate a new model regardless of preserved shapes
        model, _ = run_fitting(lmk, model=model, output_file=output_file,
                               preserve_shape=args.preserve_shape)


# -----------------------------------------------------------------------------


if __name__ == '__main__':

    main()

