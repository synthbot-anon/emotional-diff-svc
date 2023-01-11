#!/usr/bin/env python
# coding: utf-8

# In[ ]:


character_name = "sunset" # I don't know if it's okay to use spaces here
dataset_path = "/workspace/data.zip"
binary_path = "/workspace/binary.zip" # use a binary.zip file if you have one, otherwise one will be created here
model_path = "/workspace/models" # pick a directory for storing .ckpt (model) files
mount_gdrive = '' # set this to '/gdrive' if you're on Colab and want to use your gdrive
save_every = 2000 # 2000 steps = every ~10m on an RTX 3090, ~30m on RTX 3080

install_dependencies = False # set this to false if you've already successfully run this cell
install_directory = '/workspace/diff-svc' # if you're running locally, pick a directory you can modify

# after setting everything above, run this cell
# ---------------------------------------------

import os

if mount_gdrive:
    from google.colab import drive
    drive.mount(mount_gdrive, force_remount=True)

args_ok = True

if not os.path.exists(dataset_path):
    print(f"there's nothing in your dataset path ({dataset_path})")
    print(" -> please update it and run this cell again")
    args_ok = False

if args_ok:
    if not os.path.exists(os.path.dirname(install_directory)):
        os.makedirs(os.path.dirname(install_directory), exist_ok=True)
    checkpoint_dir = f"{install_directory}/checkpoints"

    # install dependencies

    if install_dependencies:
        # download prereq packages
        get_ipython().system('git clone https://github.com/synthbot-anon/emotional-diff-svc.git "{install_directory}"')
        get_ipython().system('(cd "{install_directory}"; git checkout -f cc000a254be89f467de881e6c5c48d7b9b8e590f)')
        get_ipython().system('sudo apt install -y zip libsndfile1 gcc')
        get_ipython().system('pip install torch torchvision torchaudio librosa h5py matplotlib praat-parselmouth pyloudnorm torchcrepe webrtcvad scikit-image pycwt')
        get_ipython().system('pip install -r "{install_directory}/requirements_short.txt"')
        get_ipython().system('pip install --upgrade pytorch-lightning')

        # download prereq model checkpoints
        os.mkdir(checkpoint_dir)
        os.chdir(checkpoint_dir)
        get_ipython().system('wget https://github.com/justinjohn0306/diff-svc/releases/download/models/0102_xiaoma_pe.zip')
        get_ipython().system('wget https://github.com/justinjohn0306/diff-svc/releases/download/models/hubert.zip')
        get_ipython().system('wget https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip')
        get_ipython().system('unzip 0102_xiaoma_pe.zip')
        get_ipython().system('unzip hubert.zip')
        get_ipython().system('unzip nsf_hifigan_20221211.zip')
        get_ipython().system('rm *.zip')
    
    if os.path.abspath(model_path) != os.path.abspath(checkpoint_dir):
        # leave the checkpoint path alone if that's where we want to store models
        
        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)
        
        character_path = os.path.join(checkpoint_dir, character_name)
        get_ipython().system('rm -r "{character_path}" 2> /dev/null')
        os.symlink(model_path, character_path)
    
    if not os.path.exists(os.path.dirname(binary_path)):
        os.makedirs(os.path.dirname(binary_path, exist_ok=True))
  
    print('done')
        


# In[ ]:


# preprocess the data... just run this cell

unpack_data = True #@param {type:"boolean"}
create_config = True #@param {type:"boolean"}

# create the data dir
if unpack_data:
    print('unpacking the dataset')

    data_dir = os.path.join(install_directory, 'data', 'raw', character_name)
    get_ipython().system('rm -r {data_dir} 2> /dev/null')
    os.makedirs(data_dir, exist_ok=True)

    from zipfile import ZipFile
    import soundfile
    from io import BytesIO
    import librosa

    # unpack zipped data
    modified_data = False
    mod_types = set()

    with ZipFile(dataset_path, 'r') as data:
        for info in data.infolist():
            if info.is_dir():
                continue

            fn = info.filename

            # unpack the data
            with data.open(fn) as inp:
                sound_data = inp.read()            
                try:
                    audio, sr = soundfile.read(BytesIO(sound_data))
                except:
                    print('skipping invalid file in {os.path.basename(dataset_path)}:', fn)
                    continue

                # force the new filename to match the required format
                dirname, basename = os.path.split(os.path.normpath(fn))
                dir_paths = dirname.split(os.sep)
                dir_paths = list(filter(lambda x: x not in ('', 'data', 'raw', character_name), dir_paths))
                new_dir = '-'.join(dir_paths)
                new_path = os.path.join(new_dir, basename)

                # normalize audio if needed
                if sr != 44100:
                    print('resampling', new_path)
                    audio, sr = librosa.resample(audio, sr, 44100)
                    modified_data = True
                    mod_types.add('resampled audio to 44.1 khz')

                # write to disk
                write_path = os.path.join(data_dir, new_path)
                os.makedirs(os.path.dirname(write_path), exist_ok=True)
                soundfile.write(write_path, audio, sr)

    if modified_data:
        print('the data was modified:')
        print('  -', '\n  - '.join(mod_types))
        print(f'saving a copy of the modifed data to {install_directory}/updated-data.zip')
        get_ipython().system('python3 -m zipfile -c "{install_directory}/updated-data.zip" "{data_dir}"')
        print('Hey! Please use ^ in the future.')
        print('--------------------------------')
        print('')
        print('')

    print('done unpacking the dataset')

if create_config:
    print('creating the config file... ', end='')
    import yaml
    import glob

    config_path = os.path.join(install_directory, 'training', 'config_nsf.yaml')
    with open(config_path) as inp:
        config = yaml.full_load(inp)

    config['binary_data_dir'] = f'data/binary/{character_name}'
    config['raw_data_dir'] = f'data/raw/{character_name}'
    config['speaker_id'] = character_name
    config['work_dir'] = f'checkpoints/{character_name}'
    config['val_check_interval'] = save_every

    with open(config_path, 'w') as outp:
        yaml.dump(config, outp)
    
    model_config = os.path.join(model_path, 'config.yaml')
    if glob.glob(f'{model_path}/*.ckpt') or os.path.exists(model_config):
        with open(model_config, 'w') as outp:
            yaml.dump(config, outp)

    print('done')


if os.path.exists(binary_path):
    print('using preprocessed data from', binary_path)
    get_ipython().system('python3 -m zipfile -e "{binary_path}" "{install_directory}/data/binary"')
    print('done')
else:
    print('preprocessing the data...')
    get_ipython().system('rm -r "{install_directory}/data/binary" 2> /dev/null')
    os.chdir(install_directory)
    get_ipython().system('PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python3 preprocessing/binarize.py --config training/config_nsf.yaml')
    print('packing the results... ', end='')
    get_ipython().system('python3 -m zipfile -c "{binary_path}" "{install_directory}/data/binary"')
    print('done')


# In[ ]:


os.chdir(install_directory)
if glob.glob(f"{model_path}/*.ckpt"):
    print('resuming from last checkpoint')
    get_ipython().system('CUDA_VISIBLE_DEVICES=0 python run.py --config training/config_nsf.yaml --exp_name sunset')
else:
    print('starting a new training run')
    get_ipython().system('CUDA_VISIBLE_DEVICES=0 python run.py --config training/config_nsf.yaml --exp_name sunset --reset')

