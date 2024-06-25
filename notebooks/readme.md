### Github for 3d object generation
! git clone https://github.com/threestudio-project/threestudio.git

### Installing git-lfs for downloading large files (model checkpoint)
! curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
! apt-get install git-lfs
! git lfs install

### clone repo containing model_weights.ckpt
! git clone https://huggingface.co/stabilityai/stable-zero123
copy the .ckpt file into /threestudio/load/zero123


